/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <queue>

#include "criterion/TransformerCriterion.h"

using namespace fl;

namespace w2l {

TransformerCriterion buildTransformerCriterion(
    int numClasses,
    int numLayers,
    float dropout,
    float layerdrop,
    int eosIdx) {
  std::shared_ptr<AttentionBase> attention;
  if (FLAGS_attention == w2l::kContentAttention) {
    attention = std::make_shared<ContentAttention>();
  } else if (FLAGS_attention == w2l::kKeyValueAttention) {
    attention = std::make_shared<ContentAttention>(true);
  } else if (FLAGS_attention == w2l::kNeuralContentAttention) {
    attention = std::make_shared<NeuralContentAttention>(FLAGS_encoderdim);
  } else if (FLAGS_attention == w2l::kSimpleLocationAttention) {
    attention = std::make_shared<SimpleLocationAttention>(FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kLocationAttention) {
    attention = std::make_shared<LocationAttention>(
        FLAGS_encoderdim, FLAGS_attnconvkernel);
  } else if (FLAGS_attention == w2l::kNeuralLocationAttention) {
    attention = std::make_shared<NeuralLocationAttention>(
        FLAGS_encoderdim,
        FLAGS_attndim,
        FLAGS_attnconvchannel,
        FLAGS_attnconvkernel);
  } else {
    throw std::runtime_error("Unimplmented attention: " + FLAGS_attention);
  }

  std::shared_ptr<WindowBase> window;
  if (FLAGS_attnWindow == w2l::kNoWindow) {
    window = nullptr;
  } else if (FLAGS_attnWindow == w2l::kMedianWindow) {
    window = std::make_shared<MedianWindow>(
        FLAGS_leftWindowSize, FLAGS_rightWindowSize);
  } else if (FLAGS_attnWindow == w2l::kStepWindow) {
    window = std::make_shared<StepWindow>(
        FLAGS_minsil, FLAGS_maxsil, FLAGS_minrate, FLAGS_maxrate);
  } else if (FLAGS_attnWindow == w2l::kSoftWindow) {
    window = std::make_shared<SoftWindow>(
        FLAGS_softwstd, FLAGS_softwrate, FLAGS_softwoffset);
  } else if (FLAGS_attnWindow == w2l::kSoftPretrainWindow) {
    window = std::make_shared<SoftPretrainWindow>(FLAGS_softwstd);
  } else {
    throw std::runtime_error("Unimplmented window: " + FLAGS_attnWindow);
  }

  return TransformerCriterion(
      numClasses,
      FLAGS_encoderdim,
      eosIdx,
      FLAGS_maxdecoderoutputlen,
      numLayers,
      attention,
      window,
      FLAGS_trainWithWindow,
      FLAGS_labelsmooth,
      FLAGS_pctteacherforcing,
      dropout,
      layerdrop);
}

TransformerCriterion::TransformerCriterion(
    int nClass,
    int hiddenDim,
    int eos,
    int maxDecoderOutputLen,
    int nLayer,
    std::shared_ptr<AttentionBase> attention,
    std::shared_ptr<WindowBase> window,
    bool trainWithWindow,
    double labelSmooth,
    double pctTeacherForcing,
    double p_dropout,
    double p_layerdrop)
    : nClass_(nClass),
      eos_(eos),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      nLayer_(nLayer),
      window_(window),
      trainWithWindow_(trainWithWindow),
      labelSmooth_(labelSmooth),
      pctTeacherForcing_(pctTeacherForcing) {
  add(std::make_shared<fl::Embedding>(hiddenDim, nClass));
  for (size_t i = 0; i < nLayer_; i++) {
    add(std::make_shared<Transformer>(
        hiddenDim,
        hiddenDim / 4,
        hiddenDim * 4,
        4,
        maxDecoderOutputLen,
        p_dropout,
        p_layerdrop,
        true));
  }
  add(std::make_shared<fl::Linear>(hiddenDim, nClass));
  add(attention);
  params_.push_back(fl::uniform(af::dim4{hiddenDim}, -1e-1, 1e-1));
}

Variable applyMask(
    const Variable& input,
    const af::array& mask) {
  af::array output = af::flat(input.array());
  af::array flatMask = af::flat(mask);
  auto inputDims = input.dims();

  output(flatMask) = 0;
  output = af::moddims(output, inputDims);
  
  auto gradFunc =
      [flatMask, inputDims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        af::array gradArray = af::flat(gradOutput.array());
        gradArray(flatMask) = 0.;
        auto grad = Variable(af::moddims(gradArray, inputDims), false);
        inputs[0].addGrad(grad);
      };
  return Variable(output, {input.withoutData()}, gradFunc);
}

std::vector<Variable> TransformerCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 3) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];
  const auto& inputProportions = inputs[2];

  int ignoreIndex = -1;
  if (FLAGS_padfix) {
    ignoreIndex = eos_ + 1;
  }
  
  af::array targetSizes = af::moddims(af::sum((target.array() != ignoreIndex).as(s32), 0), af::dim4(target.dims(1)));

  Variable out, alpha;
  std::tie(out, alpha) = vectorizedDecoder(input, target, inputProportions, targetSizes);

  out = logSoftmax(out, 0);
  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE, ignoreIndex), {0}), -1);
  // if (FLAGS_use_new_batching_random) {
  //   auto normalization = af::sum((af::flat(target.array()) != ignoreIndex).as(s32), 0).scalar<int>();
  //   if (normalization < 1) {
  //     losses = losses * 0;
  //   } else {
  //     losses = losses / normalization;
  //   }
  // }
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dims(0);
    if (FLAGS_padfix2) {
      auto mask = af::tile(af::moddims(target.array(), af::dim4(1, target.dims(0), target.dims(1))), nClass) == ignoreIndex;
      out = applyMask(out, mask);
    }

    auto smoothLoss = moddims(sum(out, {0, 1}), -1);
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses, out};
}

// input : D x T x B
// target: U x B

std::pair<Variable, Variable> TransformerCriterion::vectorizedDecoder(
    const Variable& input,
    const Variable& target,
    const Variable& inputProportions,
    const af::array& targetSizes) {
  int U = target.dims(0);
  int B = target.dims(1);
  int T = input.isempty() ? 0 : input.dims(1);

  auto hy = tile(startEmbedding(), {1, 1, B});

  if (U > 1) {
    auto y = target(af::seq(0, U - 2), af::span);

    if (train_) {
      // TODO: other sampling strategies
      auto mask =
          Variable(af::randu(y.dims()) * 100 <= pctTeacherForcing_, false);
      auto samples =
          Variable((af::randu(y.dims()) * (nClass_ - 1)).as(s32), false);
      y = mask * y + (1 - mask) * samples;
    }

    auto yEmbed = embedding()->forward(y);
    hy = concatenate({hy, yEmbed}, 1);
  }

  Variable alpha, summaries;
  af::array padMask; // no mask, decoder is not looking into future
  for (int i = 0; i < nLayer_; i++) {
    hy = layer(i)->forward(std::vector<Variable>({hy, fl::noGrad(padMask)})).front();
  }
  if (!input.isempty()) {
    Variable windowWeight;
    if (window_ && (!train_ || trainWithWindow_)) {
      windowWeight = window_->computeWindowMask(U, T, B, inputProportions.array(), targetSizes);
    }

    std::tie(alpha, summaries) =
        attention()->forward(hy, input, Variable(), windowWeight, inputProportions);

    hy = hy + summaries;
  }
  auto out = linearOut()->forward(hy);
  return std::make_pair(out, alpha);
}

af::array TransformerCriterion::viterbiPath(
  const af::array& input, 
  const fl::Variable& inputProportions) {
  return viterbiPathBase(input, inputProportions, false).first;
}

std::pair<af::array, Variable> TransformerCriterion::viterbiPathBase(
    const af::array& input,
    const fl::Variable& inputProportions,
    bool /* TODO: saveAttn */) {
  bool wasTrain = train_;
  eval();
  std::vector<int> path;
  std::vector<Variable> alphaVec;
  Variable alpha;
  TS2SState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;

  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    std::tie(ox, state) = decodeStep(Variable(input, false), y, state, inputProportions);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);
    // TODO: saveAttn

    if (pred == eos_) {
      break;
    }
    y = constant(pred, 1, s32, false);
    path.push_back(pred);
  }
  // TODO: saveAttn

  if (wasTrain) {
    train();
  }

  auto vPath = path.empty() ? af::array() : af::array(path.size(), path.data());
  return std::make_pair(vPath, alpha);
}

std::pair<Variable, TS2SState> TransformerCriterion::decodeStep(
    const Variable& xEncoded,
    const Variable& y,
    const TS2SState& inState,
    const Variable& inputProportions) const {
  size_t stepSize = fl::afGetMemStepSize();
  fl::afSetMemStepSize(100 * (1 << 10));

  Variable hy;
  if (y.isempty()) {
    hy = tile(startEmbedding(), {1, 1, xEncoded.dims(2)});
  } else {
    hy = embedding()->forward(y);
  }

  // TODO: inputFeeding

  TS2SState outState;
  outState.step = inState.step + 1;
  af::array padMask; // no mask because we are doing step by step decoding here
  for (int i = 0; i < nLayer_; i++) {
    if (inState.step == 0) {
      outState.hidden.push_back(hy);
      hy = layer(i)->forward(std::vector<Variable>({hy, fl::noGrad(padMask)})).front();
    } else {
      outState.hidden.push_back(concatenate({inState.hidden[i], hy}, 1));
      hy = layer(i)->forward({inState.hidden[i], hy, fl::noGrad(padMask)}).front();
    }
  }

  Variable windowWeight, alpha, summary;
  if (window_ && (!train_ || trainWithWindow_)) {
    windowWeight = window_->computeSingleStepWindow(
        Variable(), xEncoded.dims(1), xEncoded.dims(2), inState.step);
  }

  std::tie(alpha, summary) =
      attention()->forward(hy, xEncoded, Variable(), windowWeight, inputProportions);

  hy = hy + summary;

  auto out = linearOut()->forward(hy);
  fl::afSetMemStepSize(stepSize);
  return std::make_pair(out, outState);
}

std::pair<std::vector<std::vector<float>>, std::vector<TS2SStatePtr>>
TransformerCriterion::decodeBatchStep(
    const fl::Variable& xEncoded,
    std::vector<fl::Variable>& ys,
    const std::vector<TS2SState*>& inStates,
    const int /* attentionThreshold */,
    const float smoothingTemperature) const {
  size_t stepSize = fl::afGetMemStepSize();
  fl::afSetMemStepSize(10 * (1 << 10));
  int B = ys.size();

  for (int i = 0; i < B; i++) {
    if (ys[i].isempty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
    } // TODO: input feeding
    ys[i] = moddims(ys[i], {ys[i].dims(0), 1, -1});
  }
  Variable yBatched = concatenate(ys, 2); // D x 1 x B

  std::vector<TS2SStatePtr> outstates(B);
  for (int i = 0; i < B; i++) {
    outstates[i] = std::make_shared<TS2SState>();
    outstates[i]->step = inStates[i]->step + 1;
  }

  Variable outStateBatched;
  af::array padMask; // no mask because we are doing step by step decoding here
  for (int i = 0; i < nLayer_; i++) {
    if (inStates[0]->step == 0) {
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(yBatched.slice(j));
      }
      yBatched = layer(i)->forward(std::vector<Variable>({yBatched, fl::noGrad(padMask)})).front();
    } else {
      std::vector<Variable> statesVector(B);
      for (int j = 0; j < B; j++) {
        statesVector[j] = inStates[j]->hidden[i];
      }
      Variable inStateHiddenBatched = concatenate(statesVector, 2);
      auto tmp = concatenate({inStateHiddenBatched, yBatched}, 1);
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(tmp.slice(j));
      }
      yBatched = layer(i)->forward({inStateHiddenBatched, yBatched, fl::noGrad(padMask)}).front();
    }
  }

  Variable alpha, summary;
  yBatched = moddims(yBatched, {yBatched.dims(0), -1});
  std::tie(alpha, summary) =
      attention()->forward(yBatched, xEncoded, Variable(), Variable());
  alpha = reorder(alpha, 1, 0);
  yBatched = yBatched + summary;

  auto outBatched = linearOut()->forward(yBatched);
  outBatched = logSoftmax(outBatched / smoothingTemperature, 0);
  std::vector<std::vector<float>> out(B);
  for (int i = 0; i < B; i++) {
    out[i] = w2l::afToVector<float>(outBatched.col(i));
  }

  fl::afSetMemStepSize(stepSize);
  return std::make_pair(out, outstates);
}

AMUpdateFunc buildTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& c) {
  auto buf = std::make_shared<TS2SDecoderBuffer>(
      FLAGS_beamsize, FLAGS_attentionthreshold, FLAGS_smoothingtemperature);

  const TransformerCriterion* criterion =
      static_cast<TransformerCriterion*>(c.get());

  auto amUpdateFunc = [buf, criterion](
                          const float* emissions,
                          const int N,
                          const int T,
                          const std::vector<int>& rawY,
                          const std::vector<AMStatePtr>& rawPrevStates,
                          int& t) {
    if (t == 0) {
      buf->input = fl::Variable(af::array(N, T, emissions), false);
    }
    int B = rawY.size();
    buf->prevStates.resize(0);
    buf->ys.resize(0);

    for (int i = 0; i < B; i++) {
      TS2SState* prevState = static_cast<TS2SState*>(rawPrevStates[i].get());
      fl::Variable y;
      if (t > 0) {
        y = fl::constant(rawY[i], 1, s32, false);
      } else {
        prevState = &buf->dummyState;
      }
      buf->ys.push_back(y);
      buf->prevStates.push_back(prevState);
    }

    std::vector<std::vector<float>> amScores;
    std::vector<TS2SStatePtr> outStates;

    std::tie(amScores, outStates) = criterion->decodeBatchStep(
        buf->input,
        buf->ys,
        buf->prevStates,
        buf->attentionThreshold,
        buf->smoothingTemperature);

    std::vector<AMStatePtr> out;
    for (auto& os : outStates) {
      out.push_back(os);
    }

    return std::make_pair(amScores, out);
  };

  return amUpdateFunc;
}

std::string TransformerCriterion::prettyString() const {
  return "TransformerCriterion";
}

} // namespace w2l
