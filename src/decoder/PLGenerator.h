/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include "criterion/criterion.h"
#include "flashlight/common/Serialization.h"
#include "libraries/common/ProducerConsumerQueue.h"
#include "libraries/decoder/LexiconDecoder.h"
#include "libraries/decoder/LexiconFreeDecoder.h"
#include "libraries/decoder/LexiconFreeSeq2SeqDecoder.h"
#include "libraries/decoder/LexiconSeq2SeqDecoder.h"
#include "libraries/lm/ConvLM.h"
#include "libraries/lm/KenLM.h"
#include "libraries/lm/ZeroLM.h"
#include "module/module.h"
#include "runtime/runtime.h"

namespace w2l {

struct BeamElement {
  double targetLength;
  float wer;
  std::vector<std::string> trans;
  double amScore;
  double lmScore;

  FL_SAVE_LOAD(targetLength, wer, trans, amScore, lmScore)
};

class PLGenerator {
 public:
  PLGenerator(
      const Dictionary& tokenDict,
      const Dictionary& wordDict,
      const LexiconMap& lexicon,
      const std::string& runPath,
      int worldRank,
      int worldSize);

  std::shared_ptr<W2lDataset> reloadPL(
      int& startEpoch,
      const std::pair<std::string, std::string>& firstValidSet,
      const std::shared_ptr<fl::Module>& ntwrk,
      const std::shared_ptr<SequenceCriterion> criterion,
      const std::shared_ptr<W2lDataset> trainds);

  std::shared_ptr<W2lDataset> regenratePL(
      int curEpoch,
      const std::pair<std::string, std::string>& firstValidSet,
      const std::shared_ptr<fl::Module>& ntwrk,
      const std::shared_ptr<SequenceCriterion> criterion,
      const std::shared_ptr<W2lDataset> trainds);

  void setModelWER(const float& wer);

  void setRank(int worldRank, int worldSize);

  void setDictionary(
      const Dictionary& tokenDict,
      const Dictionary& wordDict,
      const LexiconMap& lexicon);

  void loadLMandTrie();

  void loadRsDictionary();

  void resetFlags(std::string runPath, int worldRank, int worldSize);

 private:
  int worldRank_;
  int worldSize_;
  int minTargetSize_;
  CriterionType criterionType_;
  std::shared_ptr<LM> lm_;
  std::shared_ptr<Trie> trie_;
  Dictionary usrDict_;
  Dictionary tokenDict_;
  LexiconMap lexicon_;
  Dictionary wordDict_;
  std::string runPath_;
  int unkWordIdx_;
  int blankIdx_;
  int silIdx_;

  std::vector<double> lmweightRange_;
  std::vector<double> wordscoreRange_;
  std::vector<double> eosscoreRange_;
  std::vector<double> lmweightList_;
  std::vector<double> wordscoreList_;
  std::vector<double> eosscoreList_;
  double bestLmWeight_;
  double besWordScore_;
  double bestEosScore_;

  double seedModelWER_;
  double currentModelWER_;

  double filteringWER_;
  double filteringPPL_{1000000000000.};

  std::vector<std::string> plEpochVec_;
  std::unordered_map<int, std::pair<int, bool>> plUpdateMap_;
  std::vector<std::string> unsupFiles_;

  std::shared_ptr<fl::Module> rsLm_;
  std::shared_ptr<fl::Module> rsCriterion_;
  Dictionary rsDict_;

  std::vector<std::vector<BeamElement>> beamVec_; // For decoding sweep

  FL_SAVE_LOAD(
      worldRank_,
      worldSize_,
      minTargetSize_,
      criterionType_,
      runPath_,
      unkWordIdx_,
      blankIdx_,
      silIdx_,
      lmweightRange_,
      wordscoreRange_,
      eosscoreRange_,
      bestLmWeight_,
      besWordScore_,
      bestEosScore_,
      seedModelWER_,
      currentModelWER_,
      filteringWER_,
      plEpochVec_,
      plUpdateMap_,
      unsupFiles_,
      rsLm_,
      rsCriterion_,
      fl::versioned(filteringPPL_, 1))

  PLGenerator() = default;

  void generateRandomWeights();

  std::vector<BeamElement> generateBeam(
      const std::vector<af::array>& sample,
      const std::shared_ptr<fl::Module>& ntwrk,
      std::shared_ptr<Decoder> decoder);

  std::shared_ptr<Decoder> buildDecoder(
      const DecoderOptions& decoderOpt,
      const std::shared_ptr<SequenceCriterion> criterion);
};
} // namespace w2l

namespace helper {
  struct ErrorState {
    int64_t ndel; //!< Number of deletion error
    int64_t nins; //!< Number of insertion error
    int64_t nsub; //!< Number of substitution error
    ErrorState() : ndel(0), nins(0), nsub(0) {}

    /** Sums up all the errors. */
    int64_t sum() const {
      return ndel + nins + nsub;
    }
  };

  template <typename T>
  double levensteinDistance(
      const T& in1begin,
      const T& in2begin,
      size_t len1,
      size_t len2) {
    std::vector<ErrorState> column(len1 + 1);
    for (int i = 0; i <= len1; ++i) {
      column[i].nins = i;
    }

    auto curin2 = in2begin;
    for (int x = 1; x <= len2; x++) {
      ErrorState lastdiagonal = column[0];
      column[0].ndel = x;
      auto curin1 = in1begin;
      for (int y = 1; y <= len1; y++) {
        auto olddiagonal = column[y];
        auto possibilities = {
            column[y].sum() + 1,
            column[y - 1].sum() + 1,
            lastdiagonal.sum() + ((*curin1 == *curin2) ? 0 : 1)};
        auto min_it =
            std::min_element(possibilities.begin(), possibilities.end());
        if (std::distance(possibilities.begin(), min_it) ==
            0) { // deletion error
          ++column[y].ndel;
        } else if (
            std::distance(possibilities.begin(), min_it) == 1) { // insertion
                                                                 // error
          column[y] = column[y - 1];
          ++column[y].nins;
        } else {
          column[y] = lastdiagonal;
          if (*curin1 != *curin2) { // substitution error
            ++column[y].nsub;
          }
        }

        lastdiagonal = olddiagonal;
        ++curin1;
      }
      ++curin2;
    }

    return column[len1].sum() / (double)len2;
  }
} // namespace helper

