/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: Please refer to https://git.io/JvJuR");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::Module> lengthNetwork;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  af::setDevice(0);
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  network->eval();
  criterion->eval();

  if (FLAGS_decoder_length_model != "") {
    std::unordered_map<std::string, std::string> dummyCfg;
    W2lSerializer::load(FLAGS_decoder_length_model, dummyCfg, lengthNetwork);
    LOG(INFO) << "Loaded length network " << FLAGS_decoder_length_model;
    lengthNetwork->eval();
  }

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  // Only Copy any values from deprecated flags to new flags when deprecated
  // flags are present and corresponding new flags aren't
  w2l::handleDeprecatedFlags();

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
  }
  Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  if (FLAGS_eostoken) {
    tokenDict.addEntry(kEosToken);
  }
  if (FLAGS_padfix) {
    tokenDict.addEntry("<PAD>");
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  Dictionary wordDict;
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
  }

  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* ===================== Create Dataset ===================== */
  // Load dataset
  auto ds = createDataset(
      FLAGS_test,
      dicts,
      lexicon,
      FLAGS_batchsize /* batchsize */,
      0 /* worldrank */,
      1 /* worldsize */);

  ds->shuffle(3);
  int nSamples = ds->size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }
  LOG(INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Test ===================== */
  std::vector<double> sliceWer(FLAGS_nthread_decoder_am_forward);
  std::vector<double> sliceLer(FLAGS_nthread_decoder_am_forward);
  std::vector<int> sliceNumWords(FLAGS_nthread_decoder_am_forward, 0);
  std::vector<int> sliceNumTokens(FLAGS_nthread_decoder_am_forward, 0);
  std::vector<int> sliceNumSamples(FLAGS_nthread_decoder_am_forward, 0);
  std::vector<double> sliceTime(FLAGS_nthread_decoder_am_forward, 0);

  auto cleanTestPath = cleanFilepath(FLAGS_test);
  std::string emissionDir;
  if (!FLAGS_emission_dir.empty()) {
    emissionDir = pathsConcat(FLAGS_emission_dir, cleanTestPath);
    dirCreate(emissionDir);
  }

  // Prepare sclite log writer
  std::ofstream hypStream, refStream;
  if (!FLAGS_sclite.empty()) {
    auto hypPath = pathsConcat(FLAGS_sclite, cleanTestPath + ".hyp");
    auto refPath = pathsConcat(FLAGS_sclite, cleanTestPath + ".viterbi.ref");
    hypStream.open(hypPath);
    refStream.open(refPath);
    if (!hypStream.is_open() || !hypStream.good()) {
      LOG(FATAL) << "Error opening hypothesis file: " << hypPath;
    }
    if (!refStream.is_open() || !refStream.good()) {
      LOG(FATAL) << "Error opening reference file: " << refPath;
    }
  }

  std::mutex hypMutex, refMutex;
  auto writeHyp = [&hypMutex, &hypStream](const std::string& hypStr) {
    std::lock_guard<std::mutex> lock(hypMutex);
    hypStream << hypStr;
  };
  auto writeRef = [&refMutex, &refStream](const std::string& refStr) {
    std::lock_guard<std::mutex> lock(refMutex);
    refStream << refStr;
  };

  // Run test
  std::mutex dataReadMutex;
  int datasetSampleId = 0; // A gloabal index for data reading

  auto run = [&dataReadMutex,
              &datasetSampleId,
              &network,
              &criterion,
              &lengthNetwork,
              &nSamples,
              &ds,
              &tokenDict,
              &wordDict,
              &writeHyp,
              &writeRef,
              &emissionDir,
              &sliceWer,
              &sliceLer,
              &sliceNumWords,
              &sliceNumTokens,
              &sliceNumSamples,
              &sliceTime](int tid) {
    // Initialize AM
    af::setDevice(tid);
    std::shared_ptr<fl::Module> localNetwork = network;
    std::shared_ptr<SequenceCriterion> localCriterion = criterion;
    if (tid != 0) {
      std::unordered_map<std::string, std::string> dummyCfg;
      W2lSerializer::load(FLAGS_am, dummyCfg, localNetwork, localCriterion);
      localNetwork->eval();
      localCriterion->eval();
    }

    TestMeters meters;
    meters.timer.resume();
    bool logging = true;
    while (datasetSampleId < nSamples) {
      std::vector<af::array> sample;
      {
        std::lock_guard<std::mutex> lock(dataReadMutex);
        sample = ds->get(datasetSampleId);
        datasetSampleId++;
      }
      // if (logging) {
      //   auto tmp = fl::input(sample[kInputIdx]);
      //   af::print("input", tmp.array());
        // int index = 0;
        // std::shared_ptr<fl::Sequential> localNetworkCast = std::dynamic_pointer_cast<fl::Sequential>(localNetwork);
        // for (auto module : localNetworkCast->modules()) {
        //   tmp = module->forward({tmp}).front();
        //   af::print(("tmp" + std::to_string(index)).c_str(), tmp.array());
        //   index++;
        // }
        // logging = false;
      // }
      // auto inp = fl::input(sample[kInputIdx]);
      // auto out1 = fl::reorder(fl::reorder(inp, 3, 0, 1, 2).row(0), 1, 2, 3, 0);
      // int index = 0;
      // std::shared_ptr<fl::Sequential> localNetworkCast = std::dynamic_pointer_cast<fl::Sequential>(localNetwork);
      //   for (auto module : localNetworkCast->modules()) {
      //     if (std::dynamic_pointer_cast<fl::LayerNorm>(module) != nullptr && index < 4) {
      //       int s = 267;
      //       if (index > 1) {
      //         s = 267 / 2;
      //       }
      //       LOG(INFO) << s << " " << out1.dims(0);
      //       auto a1 = fl::reorder(out1.rows(0, s), 1, 0);
      //       a1 = fl::reorder(a1, 1, 0);
      //       auto a2 =  fl::reorder(out1.rows(s + 1, out1.dims(0) - 1), 1, 0);
      //       a2 = fl::reorder(a2, 1, 0);
      //       auto tmp = module->forward({a1}).front();
      //       out1 = fl::concatenate({tmp, a2}, 0);
      //       index++;
      //     } else {
      //       out1 = module->forward({out1}).front();
      //     }
      //   }
      // auto out1 = localNetwork->forward({fl::input(sample[kInputIdx]).row(0)}).front();
      // auto out2 = fl::reorder(fl::reorder(inp, 3, 0, 1, 2).row(1), 1, 2, 3, 0);
      // out2 = localNetwork->forward({out2}).front();
      // auto rawEmissionBatch = fl::concatenate({out1, out2}, 2);
      af::array padMask;
      if (FLAGS_fixed_transformer) {
          int T = sample[kInputIdx].dims(0), B = sample[kInputIdx].dims(3);
          af::array inputNotPaddedSize = af::moddims(
            af::ceil(sample[kInputProportions] * T), af::dim4(1, B)); 
          padMask = af::iota(
            af::dim4(T, 1), af::dim4(1, B)) < af::tile(inputNotPaddedSize, T, 1); 
        }

      auto nn = std::dynamic_pointer_cast<fl::Sequential>(localNetwork);
        auto rawEmissionBatch = fl::input(sample[kInputIdx]);
        for (auto& module : nn->modules()) {
          auto tr = std::dynamic_pointer_cast<fl::Transformer>(module);
          if (tr != nullptr) {
            rawEmissionBatch = module->forward({rawEmissionBatch, fl::noGrad(padMask)}).front();
          } else {
            rawEmissionBatch = module->forward({rawEmissionBatch}).front();
          }
        }
      // auto rawEmissionBatch =
      //     localNetwork->forward({fl::input(sample[kInputIdx])}).front();
      // N T B
      // af::print("min pred", af::min(rawEmissionBatch.array(), 0));
      // af::print("max pred", af::max(rawEmissionBatch.array(), 0));
      // af::print("mean pred", af::mean(rawEmissionBatch.array(), 0));
      // af::print("min mean pred", af::mean(af::min(rawEmissionBatch.array(), 0), 1));
      // af::print("max mean pred", af::mean(af::max(rawEmissionBatch.array(), 0), 1));
      af::array predLength, predBlanks;
      af::array batchProportions = sample[kInputProportions];
      if (lengthNetwork) {
        // 2 x T x B
        auto result = lengthNetwork->forward({fl::input(sample[kInputIdx])}).front(); 
        af::array indices, maxTmp;
        af::max(maxTmp, indices, result.array(), 0);
        predLength = af::moddims(indices, af::dim4(indices.dims(1), indices.dims(2)));
      }
      // std::cout << "Target sizes:";
      for (int i = 0; i < rawEmissionBatch.dims(2); i++) {
        auto rawEmission = fl::reorder(fl::reorder(rawEmissionBatch, 0, 2, 1).col(i), 0, 2, 1);
        auto emission = afToVector<float>(rawEmission);
        auto tokenTarget = afToVector<int>(sample[kTargetIdx].col(i));
        auto labellen = getTargetSize(tokenTarget.data(), tokenTarget.size());
        tokenTarget.resize(labellen);
        auto wordTarget = afToVector<int>(sample[kWordIdx].col(i));
        auto sampleId = readSampleIds(sample[kSampleIdx].col(i)).front();
        auto sampleProportions = batchProportions.row(i);

        auto letterTarget = tknTarget2Ltr(tokenTarget, tokenDict);
        std::vector<std::string> wordTargetStr;// = tkn2Wrd(letterTarget);
        // std::cout << " " << wordTargetStr.size();
        if (FLAGS_uselexicon) {
          wordTargetStr = wrdIdx2Wrd(wordTarget, wordDict);
        } else {
          wordTargetStr = tkn2Wrd(letterTarget);
        }
        int predLengthSample = -1;
        if (lengthNetwork) {
          auto predLengthRaw = predLength.col(i).host<unsigned int>();
          predLengthSample = 0;
          auto predToken = predLengthRaw[0];
          for (int index = 1; index < predLength.dims(0); index++) {
            if (predLengthRaw[index] != predToken) {
              predLengthSample += predToken == 0; // count word
              predToken = predLengthRaw[index];
            }
          }
          predLengthSample += predToken == 0; // count word
          LOG(INFO) << " Length " << predLengthSample << " Target " << wordTargetStr.size();
        }

        // Tokens
        auto tokenPrediction =
            afToVector<int>(localCriterion->viterbiPath(rawEmission.array(), fl::input(sampleProportions)));
        auto letterPrediction = tknPrediction2Ltr(tokenPrediction, tokenDict);

        meters.lerSlice.add(letterPrediction, letterTarget);

        // Words
        std::vector<std::string> wrdPredictionStr = tkn2Wrd(letterPrediction);
        meters.werSlice.add(wrdPredictionStr, wordTargetStr);

        if (!FLAGS_sclite.empty()) {
          writeRef(join(" ", wordTargetStr) + " (" + sampleId + ")\n");
          writeHyp(join(" ", wrdPredictionStr) + " (" + sampleId + ")\n");
        }

        if (FLAGS_show) {
          meters.ler.reset();
          meters.wer.reset();
          meters.ler.add(letterPrediction, letterTarget);
          meters.wer.add(wrdPredictionStr, wordTargetStr);

          std::cout << "|T|: " << join(" ", letterTarget) << std::endl;
          std::cout << "|P|: " << join(" ", letterPrediction) << std::endl;
          std::cout << "[sample: " << sampleId
                    << ", WER: " << meters.wer.value()[0]
                    << "\%, LER: " << meters.ler.value()[0]
                    << "\%, total WER: " << meters.werSlice.value()[0]
                    << "\%, total LER: " << meters.lerSlice.value()[0]
                    << "\%, progress (thread " << tid << "): "
                    << static_cast<float>(datasetSampleId) / nSamples * 100
                    << "\%]" << std::endl;
        }

        /* Save emission and targets */
        int nTokens = rawEmission.dims(0);
        int nFrames = rawEmission.dims(1);
        EmissionUnit emissionUnit(emission, sampleId, nFrames, nTokens, predLengthSample);

        // Update counters
        sliceNumWords[tid] += wordTargetStr.size();
        sliceNumTokens[tid] += letterTarget.size();
        sliceNumSamples[tid]++;

        if (!emissionDir.empty()) {
          std::string savePath = pathsConcat(emissionDir, sampleId + ".bin");
          W2lSerializer::save(savePath, emissionUnit);
        }
      }
      // std::cout << std::endl;
    }

    meters.timer.stop();

    sliceWer[tid] = meters.werSlice.value()[0];
    sliceLer[tid] = meters.lerSlice.value()[0];
    sliceTime[tid] = meters.timer.value();
  };

  /* Spread threades */
  auto startThreadsAndJoin = [&run](int nThreads) {
    if (nThreads == 1) {
      run(0);
    } else if (nThreads > 1) {
      fl::ThreadPool threadPool(nThreads);
      for (int i = 0; i < nThreads; i++) {
        threadPool.enqueue(run, i);
      }
    } else {
      LOG(FATAL) << "Invalid negative FLAGS_nthread_decoder_am_forward";
    }
  };
  auto timer = fl::TimeMeter();
  timer.resume();
  startThreadsAndJoin(FLAGS_nthread_decoder_am_forward);
  timer.stop();

  int totalTokens = 0, totalWords = 0, totalSamples = 0;
  for (int i = 0; i < FLAGS_nthread_decoder_am_forward; i++) {
    totalTokens += sliceNumTokens[i];
    totalWords += sliceNumWords[i];
    totalSamples += sliceNumSamples[i];
  }
  double totalWer = 0, totalLer = 0, totalTime = 0;
  for (int i = 0; i < FLAGS_nthread_decoder_am_forward; i++) {
    totalWer += sliceWer[i] * sliceNumWords[i] / totalWords;
    totalLer += sliceLer[i] * sliceNumTokens[i] / totalTokens;
    totalTime += sliceTime[i];
  }

  LOG(INFO) << "------";
  LOG(INFO) << "[Test " << FLAGS_test << " (" << totalSamples << " samples) in "
            << timer.value() << "s (actual decoding time "
            << std::setprecision(3) << totalTime / totalSamples
            << "s/sample) -- WER: " << std::setprecision(6) << totalWer
            << ", LER: " << totalLer << "]" << std::endl;

  return 0;
}
