/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cstdlib>
#include <future>

#include "decoder/PLGenerator.h"

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <string>

namespace w2l {

std::vector<float> getLmScore(
    const std::vector<std::vector<std::string>>& sentences,
    std::shared_ptr<fl::Module> lm,
    std::shared_ptr<fl::Module> criterion,
    const Dictionary& dict) {
  int64_t length = 0, batchsize = sentences.size();
  for (const auto& sentence : sentences) {
    length = std::max<int64_t>(length, sentence.size());
  }
  std::vector<int> batch(batchsize * (length + 2), 1); // pad
  for (int64_t s = 0; s < batchsize; ++s) {
    int64_t start = s * (length + 2);
    batch[start] = 2; // eos
    auto indices = dict.mapEntriesToIndices(sentences[s]);
    for (int i = 0; i < indices.size(); ++i) {
      batch[start + i + 1] = indices[i];
    }
    batch[start + indices.size() + 1] = 2; // eos
  }
  af::array arr(length + 2, batchsize, batch.data());
  af::array iarr = arr(af::seq(0, length), af::span);
  af::array tarr = arr(af::seq(1, length + 1), af::span);

  auto input = fl::Variable(iarr, false);
  auto target = fl::Variable(tarr, false);

  auto output = lm->forward({input}).front();
  auto loss = criterion->forward({output, target}).front();
  loss = sum(loss, {0});
  return afToVector<float>(-loss.array());
}

/* TODO list
 * 1. [Synchronization] Remove sleep
 * 2. [Dataset] Decouple it from glabal flags
 * 3. [Dataset] Let it return everything, including path and audio length,
 *    remove 3.2
 * 4. [Feature - done] switch model based on dev-wer
 * 5. [Feature - done] Support customize LM sweep epoch
 * 6. [Feature] continue PL generating if failed/stopped in the middle
 */

PLGenerator::PLGenerator(
    const Dictionary& tokenDict,
    const Dictionary& wordDict,
    const LexiconMap& lexicon,
    const std::string& runPath,
    int worldRank,
    int worldSize)
    : worldRank_(worldRank),
      worldSize_(worldSize),
      minTargetSize_(FLAGS_mintsz),
      tokenDict_(tokenDict),
      lexicon_(lexicon),
      wordDict_(wordDict),
      runPath_(runPath),
      seedModelWER_(FLAGS_seed_model_wer) {
  /* 0. Parse PL flags */
  dirCreate(pathsConcat(runPath_, "generated_pl"));

  // 0.1 Load PL generating intervals
  plEpochVec_ = split(',', FLAGS_pl_epoch, true);
  auto nPlFileVec = split(',', FLAGS_n_pl_file, true);
  auto decoderSweepEpochVec = split(',', FLAGS_decoder_sweep_epoch, true);
  auto lmweightScheduleVec = split(',', FLAGS_lmweight_schedule, true);
  if (plEpochVec_.size() != nPlFileVec.size()) {
    LOG(FATAL) << "plEpochVec wtf?";
  }
  if (decoderSweepEpochVec.size() != lmweightScheduleVec.size()) {
    LOG(FATAL) << "lmweightScheduleVec wtf?";
  }
  for (int i = 0; i < plEpochVec_.size(); i++) {
    plUpdateMap_[stoi(plEpochVec_[i])] = {stoi(nPlFileVec[i]), -1.};
  }
  for (int i = 0; i < decoderSweepEpochVec.size(); i++) {
    int epoch = stoi(decoderSweepEpochVec[i]);
    float lmweight = stof(lmweightScheduleVec[i]);
    if (plUpdateMap_.find(epoch) == plUpdateMap_.end()) {
      LOG(FATAL) << "decoderSweepEpochVec wtf?";
    }
    plUpdateMap_[epoch].second = lmweight;
  }
  LOG_MASTER(INFO) << "---plUpdateMap";
  for (const auto& item : plUpdateMap_) {
    LOG_MASTER(INFO) << item.first << " [" << item.second.first << ", "
                     << item.second.second << "]";
  }
  LOG_MASTER(INFO) << "---";

  unsupFiles_ = split(',', FLAGS_train_unsup, true);

  // 0.2 Load parameter range
  auto rangeVec = split(',', FLAGS_lmweight_range);
  if (rangeVec.size() != 2) {
    LOG(FATAL) << "FLAGS_lmweight_range needs two numbers";
  }
  lmweightRange_.resize(2);
  for (int i = 0; i < rangeVec.size(); i++) {
    lmweightRange_[i] = stod(rangeVec[i]);
  }

  rangeVec = split(',', FLAGS_wordscore_range);
  if (rangeVec.size() != 2) {
    LOG(FATAL) << "FLAGS_wordscore_range needs two numbers";
  }
  wordscoreRange_.resize(2);
  for (int i = 0; i < rangeVec.size(); i++) {
    wordscoreRange_[i] = stod(rangeVec[i]);
  }

  rangeVec = split(',', FLAGS_eosscore_range);
  if (rangeVec.size() != 2) {
    LOG(FATAL) << "FLAGS_eosscore_range needs two numbers";
  }
  eosscoreRange_.resize(2);
  for (int i = 0; i < rangeVec.size(); i++) {
    eosscoreRange_[i] = stod(rangeVec[i]);
  }

  bestLmWeight_ = -1;
  besWordScore_ = -1;
  bestEosScore_ = -1;

  /* 1. Prepare criterion */
  criterionType_ = CriterionType::ASG;
  if (FLAGS_criterion == kCtcCriterion) {
    criterionType_ = CriterionType::CTC;
  } else if (
      FLAGS_criterion == kSeq2SeqCriterion ||
      FLAGS_criterion == kTransformerCriterion) {
    criterionType_ = CriterionType::S2S;
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion;
  }

  /* 2. Build Language Model and Trie */
  unkWordIdx_ = -1;
  usrDict_ = tokenDict;
  if (!FLAGS_lm.empty() && FLAGS_decodertype == "wrd") {
    usrDict_ = wordDict;
    unkWordIdx_ = wordDict.getIndex(kUnkToken);
  }

  blankIdx_ =
      FLAGS_criterion == kCtcCriterion ? tokenDict_.getIndex(kBlankToken) : -1;
  silIdx_ = tokenDict_.getIndex(FLAGS_wordseparator);
  trie_ = nullptr;

  loadLMandTrie();

  /* 3. TR LM */
  if (FLAGS_is_rescore) {
    W2lSerializer::load(FLAGS_tr_lm, rsLm_, rsCriterion_);
    rsLm_->eval();
    rsCriterion_->eval();
    auto tr_adsm = dynamic_cast<fl::AdaptiveSoftMaxLoss*>(rsCriterion_.get());
    auto softmax = tr_adsm->getActivation();
    rsCriterion_ = std::make_shared<fl::AdaptiveSoftMaxLoss>(
        softmax, fl::ReduceMode::NONE, 1);
    rsCriterion_->eval();

    loadRsDictionary();
  }
}

std::shared_ptr<W2lDataset> PLGenerator::reloadPL(
    int& startEpoch,
    const std::pair<std::string, std::string>& firstValidSet,
    const std::shared_ptr<fl::Module>& ntwrk,
    const std::shared_ptr<SequenceCriterion> criterion,
    const std::shared_ptr<W2lDataset> trainds) {
  // Re-initialize train set for continue mode
  int old_pl_epoch = -1;
  for (const auto& i : plEpochVec_) {
    int j = stoi(i);
    if (j > startEpoch) {
      break;
    }
    old_pl_epoch = j;
  }
  if (old_pl_epoch <= 0) {
    return trainds;
  }

  std::string plDir = pathsConcat(
      runPath_, "generated_pl/epoch_" + std::to_string(old_pl_epoch));
  auto selectListsPath = pathsConcat(plDir, "unsup.lst");

  DictionaryMap dicts = {{kTargetIdx, tokenDict_}, {kWordIdx, wordDict_}};

  if (FLAGS_use_existing_pl && seedModelWER_ < currentModelWER_) {
    LOG(INFO) << "Using existing PL (seed " << seedModelWER_ << ", current "
              << currentModelWER_ << ")";

    if (!fileExists(selectListsPath)) {
      LOG(FATAL) << selectListsPath << " doesn't exists";
    }

    std::string unsupFileList;
    std::ifstream selectListsStream(selectListsPath);
    selectListsStream >> unsupFileList;
    LOG_MASTER(INFO) << "Using " << unsupFileList;
    selectListsStream.close();
    FLAGS_mintsz = minTargetSize_;
    return createDataset(
        FLAGS_train + "," + unsupFileList,
        dicts,
        lexicon_,
        FLAGS_batchsize,
        worldRank_,
        worldSize_);
  }

  auto newTrainList = FLAGS_train;
  bool isPLReady = true;
  for (int i = 1; i < worldSize_; i++) {
    auto listFinishPath = pathsConcat(plDir, std::to_string(i) + ".fns");
    if (!fileExists(listFinishPath)) {
      isPLReady = false;
      break;
    }
    newTrainList += "," + pathsConcat(plDir, std::to_string(i) + ".lst");
  }
  if (isPLReady) {
    FLAGS_mintsz = minTargetSize_;
    return createDataset(
        newTrainList, dicts, lexicon_, FLAGS_batchsize, worldRank_, worldSize_);
  } else {
    startEpoch = old_pl_epoch;
    return regenratePL(startEpoch, firstValidSet, ntwrk, criterion, trainds);
  }
}

std::shared_ptr<W2lDataset> PLGenerator::regenratePL(
    int curEpoch,
    const std::pair<std::string, std::string>& firstValidSet,
    const std::shared_ptr<fl::Module>& ntwrk,
    const std::shared_ptr<SequenceCriterion> criterion,
    const std::shared_ptr<W2lDataset> trainds) {
  if (plUpdateMap_.find(curEpoch) == plUpdateMap_.end()) {
    return trainds;
  }

  LOG_MASTER(INFO) << "Regenerating pseudo labels";
  std::string plDir =
      pathsConcat(runPath_, "generated_pl/epoch_" + std::to_string(curEpoch));
  dirCreate(plDir);
  auto selectListsPath = pathsConcat(plDir, "unsup.lst");
  auto paramPath = pathsConcat(plDir, "param.lst");

  DictionaryMap dicts = {{kTargetIdx, tokenDict_}, {kWordIdx, wordDict_}};

  /* 1. select data */
  if (worldRank_ == 0) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(unsupFiles_.begin(), unsupFiles_.end(), g);
    int nFiles = plUpdateMap_[curEpoch].first;
    auto unsupFileList = join(
        ",",
        std::vector<std::string>(
            unsupFiles_.begin(), unsupFiles_.begin() + nFiles));

    LOG_MASTER(INFO) << "Using PL lists: " << unsupFileList;
    std::ofstream stream(selectListsPath);
    stream << unsupFileList;
    stream.close();
  }

  LOG(INFO) << "Trying to use existing PL " << FLAGS_use_existing_pl
            << " (seed " << seedModelWER_ << ", current " << currentModelWER_
            << ")";
  if (FLAGS_use_existing_pl && seedModelWER_ < currentModelWER_) {
    LOG(INFO) << "Using existing PL (seed " << seedModelWER_ << ", current "
              << currentModelWER_ << ")";
    while (!fileExists(selectListsPath)) {
      sleep(5);
    }
    sleep(5);

    std::string unsupFileList;
    std::ifstream selectListsStream(selectListsPath);
    selectListsStream >> unsupFileList;
    selectListsStream.close();
    LOG_MASTER(INFO) << "Using " << unsupFileList;
    FLAGS_mintsz = minTargetSize_;
    return createDataset(
        FLAGS_train + "," + unsupFileList,
        dicts,
        lexicon_,
        FLAGS_batchsize,
        worldRank_,
        worldSize_);
  }

  /* 2. decoder sweep */
  af::setDevice(worldRank_ % 8);

  DecoderOptions opt(
      FLAGS_beamsize,
      FLAGS_beamsizetoken,
      FLAGS_beamthreshold,
      FLAGS_lmweight,
      FLAGS_wordscore,
      FLAGS_unkscore,
      FLAGS_silscore,
      FLAGS_eosscore,
      FLAGS_logadd,
      criterionType_);
  auto decoder = buildDecoder(opt, criterion);

  if (worldRank_ == 0) {
    LOG_MASTER(INFO) << "bestLmWeight_ " << bestLmWeight_;
    LOG_MASTER(INFO) << "decoding sweep = "
                     << ((plUpdateMap_[curEpoch].second > 0 || bestLmWeight_ < 0)
                             ? "true"
                             : "false");
    if (plUpdateMap_[curEpoch].second > 0 || bestLmWeight_ < 0) {
      ntwrk->eval();
      auto ds = createDataset(firstValidSet.second, dicts, lexicon_, 1, 0, 1);

      beamVec_.clear();
      for (const auto& sample : *ds) {
        beamVec_.push_back(generateBeam(sample, ntwrk, decoder));
      }

      generateRandomWeights();
      float bestWer = 1e10;
      for (int i = 0; i < FLAGS_nthread_decoder; i++) {
        // auto lmw = lmweightList_[i];
        auto lmw = plUpdateMap_[curEpoch].second;
        auto ws = wordscoreList_[i];

        float sum = 0, weight = 0;
        for (auto& beam : beamVec_) {
          std::sort(
              beam.begin(),
              beam.end(),
              [lmw, ws](const BeamElement& p1, const BeamElement& p2) {
                double score1 =
                    p1.amScore + lmw * p1.lmScore + ws * p1.trans.size();
                double score2 =
                    p2.amScore + lmw * p2.lmScore + ws * p2.trans.size();
                return score1 > score2;
              });

          sum += beam[0].wer * beam[0].targetLength;
          weight += beam[0].targetLength;
        }

        float wer = sum / weight;
        if (wer < bestWer) {
          bestWer = wer;
          bestLmWeight_ = lmw;
          besWordScore_ = ws;
        }
      }

      std::ofstream paramStream(paramPath);
      paramStream << bestLmWeight_ << " " << besWordScore_ << " " << bestWer;
      paramStream.close();

      LOG(INFO) << "Sweep finished with: lmweight - " << bestLmWeight_
                << ", wordscore - " << besWordScore_ << ". WER: " << bestWer;
    }
  }

  /* 3. pseudo label generation */
  else {
    // 3.0 create dataset
    while (!fileExists(selectListsPath)) {
      sleep(5);
    }
    sleep(5);
    std::string unsupFileList;
    std::ifstream selectListsStream(selectListsPath);
    selectListsStream >> unsupFileList;
    LOG_MASTER(INFO) << "Using " << unsupFileList;
    selectListsStream.close();

    FLAGS_mintsz = -1;
    std::shared_ptr<W2lDataset> trainunsupds = createDataset(
        unsupFileList, dicts, lexicon_, 1, worldRank_ - 1, worldSize_ - 1);

    // 3.1 generate beam
    af::setDevice(worldRank_ % 8);

    LOG(INFO) << "Emission generating in " << worldRank_;
    for (auto& sample : *trainunsupds) {
      auto beam = generateBeam(sample, ntwrk, decoder);
      auto sampleId = readSampleIds(sample[kSampleIdx]).front();

      std::string savePath = pathsConcat(FLAGS_emission_dir, sampleId + ".bin");
      W2lSerializer::save(savePath, beam);
    }
    LOG(INFO) << "Emission generating finished in " << worldRank_;

    // 3.2 load meta data again
    std::unordered_map<std::string, std::string> metaInfo;
    for (const auto& f : unsupFiles_) {
      auto fullpath = pathsConcat(FLAGS_datadir, trim(f));
      std::ifstream infile(fullpath);
      std::string line;
      while (std::getline(infile, line)) {
        auto tokens = splitOnWhitespace(line, true);
        metaInfo[tokens[0]] = tokens[1] + " " + tokens[2];
      }
    }

    // 3.3 load decoder parameters
    if (plUpdateMap_[curEpoch].second > 0 || bestLmWeight_ < 0) {
      while (!fileExists(paramPath)) {
        sleep(5);
      }
      sleep(5);

      std::ifstream paramStream(paramPath);
      float bestWer = -1;
      paramStream >> bestLmWeight_ >> besWordScore_ >> bestWer;
      paramStream.close();

      LOG(INFO) << "Best parameter: lmweight - " << bestLmWeight_
                << ", wordscore - " << besWordScore_ << ". WER: " << bestWer;
    }

    // 3.4 decode unlabeled data + write
    LOG(INFO) << "PL generating in " << worldRank_;
    auto newListPath = pathsConcat(plDir, std::to_string(worldRank_) + ".lst");
    auto newListFinishPath =
        pathsConcat(plDir, std::to_string(worldRank_) + ".fns");
    std::ofstream nlStream(newListPath);

    for (auto& sample : *trainunsupds) {
      auto sampleId = readSampleIds(sample[kSampleIdx]).front();

      std::vector<BeamElement> beam;
      std::string savePath = pathsConcat(FLAGS_emission_dir, sampleId + ".bin");
      W2lSerializer::load(savePath, beam);
      std::remove(savePath.c_str());

      std::sort(
          beam.begin(),
          beam.end(),
          [this](const BeamElement& p1, const BeamElement& p2) {
            double score1 = p1.amScore + bestLmWeight_ * p1.lmScore +
                besWordScore_ * p1.trans.size();
            double score2 = p2.amScore + bestLmWeight_ * p2.lmScore +
                besWordScore_ * p2.trans.size();
            return score1 > score2;
          });

      nlStream << sampleId << " " << metaInfo[sampleId] << " "
               << join(" ", beam[0].trans) << std::endl;
    }

    // PL generation finished
    std::ofstream nlfStream(newListFinishPath);
    nlfStream << "done";
    nlfStream.close();
  }

  /* 4. Update train set */
  FLAGS_mintsz = minTargetSize_;
  auto newTrainList = FLAGS_train;
  for (int i = 1; i < worldSize_; i++) {
    auto listFinishPath = pathsConcat(plDir, std::to_string(i) + ".fns");
    while (!fileExists(listFinishPath)) {
      sleep(5);
    }
    newTrainList += "," + pathsConcat(plDir, std::to_string(i) + ".lst");
  }
  return createDataset(
      newTrainList, dicts, lexicon_, FLAGS_batchsize, worldRank_, worldSize_);
}

std::shared_ptr<Decoder> PLGenerator::buildDecoder(
    const DecoderOptions& decoderOpt,
    std::shared_ptr<SequenceCriterion> criterion) {
  criterion->eval();
  std::vector<float> transition;
  if (FLAGS_criterion == kAsgCriterion) {
    transition = afToVector<float>(criterion->param(0).array());
  }

  std::shared_ptr<Decoder> decoder;
  if (criterionType_ == CriterionType::S2S) {
    auto amUpdateFunc = FLAGS_criterion == kSeq2SeqCriterion
        ? buildAmUpdateFunction(criterion)
        : buildTransformerAmUpdateFunction(criterion);
    int eosIdx = tokenDict_.getIndex(kEosToken);

    if (FLAGS_decodertype == "wrd") {
      decoder.reset(new LexiconSeq2SeqDecoder(
          decoderOpt,
          trie_,
          lm_,
          eosIdx,
          amUpdateFunc,
          FLAGS_maxdecoderoutputlen,
          false));
      LOG_MASTER(INFO)
          << "[Decoder] LexiconSeq2Seq decoder with word-LM loaded";
    } else if (FLAGS_decodertype == "tkn") {
      if (FLAGS_uselexicon) {
        decoder.reset(new LexiconSeq2SeqDecoder(
            decoderOpt,
            trie_,
            lm_,
            eosIdx,
            amUpdateFunc,
            FLAGS_maxdecoderoutputlen,
            true));
        LOG_MASTER(INFO)
            << "[Decoder] LexiconSeq2Seq decoder with token-LM loaded";
      } else {
        decoder.reset(new LexiconFreeSeq2SeqDecoder(
            decoderOpt, lm_, eosIdx, amUpdateFunc, FLAGS_maxdecoderoutputlen));
        LOG_MASTER(INFO)
            << "[Decoder] LexiconFreeSeq2Seq decoder with token-LM loaded";
      }
    } else {
      LOG(FATAL) << "Unsupported decoder type: " << FLAGS_decodertype;
    }
  } else {
    if (FLAGS_decodertype == "wrd") {
      decoder.reset(new LexiconDecoder(
          decoderOpt,
          trie_,
          lm_,
          silIdx_,
          blankIdx_,
          unkWordIdx_,
          transition,
          false));
      LOG_MASTER(INFO) << "[Decoder] Lexicon decoder with word-LM loaded";
    } else if (FLAGS_decodertype == "tkn") {
      if (FLAGS_uselexicon) {
        decoder.reset(new LexiconDecoder(
            decoderOpt,
            trie_,
            lm_,
            silIdx_,
            blankIdx_,
            unkWordIdx_,
            transition,
            true));
        LOG_MASTER(INFO) << "[Decoder] Lexicon decoder with token-LM loaded";
      } else {
        decoder.reset(new LexiconFreeDecoder(
            decoderOpt, lm_, silIdx_, blankIdx_, transition));
        LOG_MASTER(INFO)
            << "[Decoder] Lexicon-free decoder with token-LM loaded";
      }
    } else {
      LOG(FATAL) << "Unsupported decoder type: " << FLAGS_decodertype;
    }
  }

  return decoder;
};

void PLGenerator::generateRandomWeights() {
  std::srand(worldRank_ + (int)(currentModelWER_ * 10000));
  lmweightList_.clear();
  wordscoreList_.clear();
  eosscoreList_.clear();
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    lmweightList_.emplace_back(
        lmweightRange_[0] +
        static_cast<double>(std::rand()) /
            (RAND_MAX / (lmweightRange_[1] - lmweightRange_[0])));
    wordscoreList_.emplace_back(
        wordscoreRange_[0] +
        static_cast<double>(std::rand()) /
            (RAND_MAX / (wordscoreRange_[1] - wordscoreRange_[0])));
    eosscoreList_.emplace_back(
        eosscoreRange_[0] +
        static_cast<double>(std::rand()) /
            (RAND_MAX / (eosscoreRange_[1] - eosscoreRange_[0])));
  }
}

std::vector<BeamElement> PLGenerator::generateBeam(
    const std::vector<af::array>& sample,
    const std::shared_ptr<fl::Module>& ntwrk,
    std::shared_ptr<Decoder> decoder) {
  // 1. Load Emissions
  auto rawEmission = ntwrk->forward({fl::input(sample[kInputIdx])}).front();
  auto emission = afToVector<float>(rawEmission);
  int N = rawEmission.dims(0);
  int T = rawEmission.dims(1);

  // 2. Load Targets
  auto tokenTarget = afToVector<int>(sample[kTargetIdx]);
  auto wordTarget = afToVector<int>(sample[kWordIdx]);
  // TODO: we will reform the w2l dataset so that the loaded word
  // targets are strings already
  std::vector<std::string> wordTargetStr;
  if (FLAGS_uselexicon) {
    wordTargetStr = wrdIdx2Wrd(wordTarget, wordDict_);
  } else {
    auto letterTarget = tknTarget2Ltr(tokenTarget, tokenDict_);
    wordTargetStr = tkn2Wrd(letterTarget);
  }

  // 3. Decode
  const auto& beam = decoder->decode(emission.data(), T, N);

  std::vector<BeamElement> res;
  int nSamples = std::min((int)FLAGS_tr_nbest, (int)beam.size());
  for (int i = 0; i < nSamples; i += FLAGS_tr_batchsize) {
    int batchsize = std::min((int)nSamples - i, (int)FLAGS_tr_batchsize);
    std::vector<std::vector<std::string>> batch;
    for (int j = i; j < i + batchsize; j++) {
      auto rawWordPrediction = beam[j].words;
      rawWordPrediction =
          validateIdx(rawWordPrediction, wordDict_.getIndex(kUnkToken));
      auto wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict_);
      batch.push_back(wordPrediction);

      TestMeters localMeters;
      localMeters.werSlice.add(wordPrediction, wordTargetStr);

      BeamElement element;
      element.targetLength = wordTargetStr.size(); // TODO: letter length
      element.trans = wordPrediction; // TODO: letter length
      element.amScore = beam[j].amScore;
      element.wer = localMeters.werSlice.value()[0];

      res.push_back(element);
    }
    auto lm_scores = getLmScore(batch, rsLm_, rsCriterion_, rsDict_);
    for (int j = i; j < i + batchsize; j++) {
      res[j].lmScore = lm_scores[j - i];
    }
  }
  return res;
}

void PLGenerator::setModelWER(const float& wer) {
  currentModelWER_ = wer;
}

void PLGenerator::setRank(int worldRank, int worldSize) {
  worldRank_ = worldRank;
  worldSize_ = worldSize;
}

void PLGenerator::setDictionary(
    const Dictionary& tokenDict,
    const Dictionary& wordDict,
    const LexiconMap& lexicon) {
  tokenDict_ = tokenDict;
  wordDict_ = wordDict;
  lexicon_ = lexicon;
  usrDict_ = tokenDict;
  if (!FLAGS_lm.empty() && FLAGS_decodertype == "wrd") {
    usrDict_ = wordDict;
    unkWordIdx_ = wordDict.getIndex(kUnkToken);
  }
}

void PLGenerator::loadLMandTrie() {
  // Build LM
  lm_ = std::make_shared<ZeroLM>();
  if (!FLAGS_lm.empty()) {
    if (FLAGS_lmtype == "kenlm") {
      lm_ = std::make_shared<KenLM>(FLAGS_lm, usrDict_);
      if (!lm_) {
        LOG(FATAL) << "[LM constructing] Failed to load LM: " << FLAGS_lm;
      }
    } else if (FLAGS_lmtype == "convlm") {
      LOG_MASTER(INFO) << "[ConvLM]: Loading LM from " << FLAGS_lm;
      std::shared_ptr<fl::Module> convLmModel;
      W2lSerializer::load(FLAGS_lm, convLmModel);
      convLmModel->eval();

      auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
      lm_ = std::make_shared<ConvLM>(
          getConvLmScoreFunc,
          FLAGS_lm_vocab,
          usrDict_,
          FLAGS_lm_memory,
          FLAGS_beamsize);
    } else {
      LOG(FATAL) << "[LM constructing] Invalid LM Type: " << FLAGS_lmtype;
    }
  }
  LOG_MASTER(INFO) << "[Decoder] LM constructed.\n";

  // Build Trie
  if (FLAGS_decodertype == "wrd" || FLAGS_uselexicon) {
    trie_ = std::make_shared<Trie>(tokenDict_.indexSize(), silIdx_);
    auto startState = lm_->start(false);

    for (auto& it : lexicon_) {
      const std::string& word = it.first;
      int usrIdx = wordDict_.getIndex(word);
      float score = -1;
      if (FLAGS_decodertype == "wrd") {
        LMStatePtr dummyState;
        std::tie(dummyState, score) = lm_->score(startState, usrIdx);
      }
      for (auto& tokens : it.second) {
        auto tokensTensor = tkn2Idx(tokens, tokenDict_, FLAGS_replabel);
        trie_->insert(tokensTensor, usrIdx, score);
      }
    }
    LOG_MASTER(INFO) << "[Decoder] Trie planted.\n";

    // Smearing
    SmearingMode smear_mode = SmearingMode::NONE;
    if (FLAGS_smearing == "logadd") {
      smear_mode = SmearingMode::LOGADD;
    } else if (FLAGS_smearing == "max") {
      smear_mode = SmearingMode::MAX;
    } else if (FLAGS_smearing != "none") {
      LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
    }
    trie_->smear(smear_mode);
    LOG_MASTER(INFO) << "[Decoder] Trie smeared.\n";
  }
}

void PLGenerator::loadRsDictionary() {
  rsDict_ = Dictionary();

  rsDict_.addEntry("<FL_DICT>");
  rsDict_.addEntry("<PAD>");
  rsDict_.addEntry("<EOS>");
  rsDict_.addEntry("<UNK>");

  rsDict_.setDefaultIndex(3);

  std::ifstream file(FLAGS_lm_vocab);
  if (!file.is_open()) {
    throw std::runtime_error(
        "failed to open file for reading: " + FLAGS_lexicon);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    rsDict_.addEntry(tkns.front());
  }
  if (!rsDict_.isContiguous()) {
    throw std::runtime_error("Invalid dictionary format - not contiguous");
  }
}

} // namespace w2l
