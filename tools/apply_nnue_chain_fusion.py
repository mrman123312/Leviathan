#!/usr/bin/env python3
from pathlib import Path

p = Path('src/nnue/nnue_accumulator.cpp')
s = p.read_text()

decl = '''void update_accumulator_incremental_both(const FeatureTransformer& featureTransformer,
                                         Square                    white_ksq,
                                         Square                    black_ksq,
                                         AccumulatorState&         target_state,
                                         const AccumulatorState&   computed);'''
decl2 = decl + '''

void update_accumulator_incremental_both_chain2(
  const FeatureTransformer& featureTransformer, Square white_ksq, Square black_ksq,
  AccumulatorState& middle_state, AccumulatorState& target_state,
  const AccumulatorState& computed);'''
if s.count(decl) != 1:
    raise SystemExit('declaration anchor')
s = s.replace(decl, decl2, 1)

loop = '''    for (usize next = shared_begin + 1; next < size; ++next)
        update_accumulator_incremental_both(featureTransformer, white_ksq, black_ksq,
                                            accumulators[next], accumulators[next - 1]);'''
loop2 = '''    for (usize next = shared_begin + 1; next < size;)
    {
        if (next + 1 < size)
        {
            update_accumulator_incremental_both_chain2(
              featureTransformer, white_ksq, black_ksq, accumulators[next],
              accumulators[next + 1], accumulators[next - 1]);
            next += 2;
        }
        else
        {
            update_accumulator_incremental_both(featureTransformer, white_ksq, black_ksq,
                                                accumulators[next], accumulators[next - 1]);
            ++next;
        }
    }'''
if s.count(loop) != 1:
    raise SystemExit('common suffix loop anchor')
s = s.replace(loop, loop2, 1)

anchor = '''void update_accumulator_incremental_both(const FeatureTransformer& featureTransformer,
                                         Square                    white_ksq,
                                         Square                    black_ksq,
                                         AccumulatorState&         target_state,
                                         const AccumulatorState&   computed) {'''

helper = r'''void update_accumulator_incremental_both_chain2(
  const FeatureTransformer& featureTransformer, Square white_ksq, Square black_ksq,
  AccumulatorState& middle_state, AccumulatorState& target_state,
  const AccumulatorState& computed) {

    assert(computed.computed[WHITE] && computed.computed[BLACK]);
    assert(!middle_state.computed[WHITE] && !middle_state.computed[BLACK]);
    assert(!target_state.computed[WHITE] && !target_state.computed[BLACK]);

#ifndef VECTOR
    update_accumulator_incremental_both(featureTransformer, white_ksq, black_ksq,
                                        middle_state, computed);
    update_accumulator_incremental_both(featureTransformer, white_ksq, black_ksq,
                                        target_state, middle_state);
#else
    PSQFeatureSet::IndexList    psq_removed[2][COLOR_NB], psq_added[2][COLOR_NB];
    ThreatFeatureSet::IndexList thr_removed[2][COLOR_NB], thr_added[2][COLOR_NB];
    const auto* threat_pp_base = &featureTransformer.threatAndPpWeights[0];
    const auto  pf_stride      = FeatureTransformer::OutputDimensions;

    auto collect = [&](int step, AccumulatorState& st) {
        ThreatFeatureSet::append_changed_indices_both(
          white_ksq, black_ksq, st.dirtyThreats,
          thr_removed[step][WHITE], thr_added[step][WHITE],
          thr_removed[step][BLACK], thr_added[step][BLACK], threat_pp_base, pf_stride);
        PairFeatureSet::append_changed_indices_both(
          white_ksq, black_ksq, st.dirtyPawnPairs,
          thr_removed[step][WHITE], thr_added[step][WHITE],
          thr_removed[step][BLACK], thr_added[step][BLACK], threat_pp_base, pf_stride);
        PSQFeatureSet::append_changed_indices(WHITE, white_ksq, st.dirtyPiece,
                                              psq_removed[step][WHITE], psq_added[step][WHITE]);
        PSQFeatureSet::append_changed_indices(BLACK, black_ksq, st.dirtyPiece,
                                              psq_removed[step][BLACK], psq_added[step][BLACK]);
    };
    collect(0, middle_state);
    collect(1, target_state);

    for (Color c : {WHITE, BLACK})
    {
        const auto& fromAcc = computed.accumulation[c];
        auto& midAcc = middle_state.accumulation[c];
        auto& toAcc  = target_state.accumulation[c];

        vec_t acc[Tiling::NumRegs];
        for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
        {
            const usize tileOff = j * Tiling::TileHeight;
            auto* fromTile = reinterpret_cast<const vec_t*>(&fromAcc[tileOff]);
            auto* midTile  = reinterpret_cast<vec_t*>(&midAcc[tileOff]);
            auto* toTile   = reinterpret_cast<vec_t*>(&toAcc[tileOff]);
            for (IndexType k = 0; k < Tiling::NumRegs; ++k) acc[k] = fromTile[k];

            apply_psq_features<-1>(j, acc, psq_removed[0][c], featureTransformer);
            apply_psq_features<+1>(j, acc, psq_added[0][c], featureTransformer);
            apply_threat_features<-1>(j, acc, thr_removed[0][c], featureTransformer);
            apply_threat_features<+1>(j, acc, thr_added[0][c], featureTransformer);
            for (IndexType k = 0; k < Tiling::NumRegs; ++k) vec_store(&midTile[k], acc[k]);

            apply_psq_features<-1>(j, acc, psq_removed[1][c], featureTransformer);
            apply_psq_features<+1>(j, acc, psq_added[1][c], featureTransformer);
            apply_threat_features<-1>(j, acc, thr_removed[1][c], featureTransformer);
            apply_threat_features<+1>(j, acc, thr_added[1][c], featureTransformer);
            for (IndexType k = 0; k < Tiling::NumRegs; ++k) vec_store(&toTile[k], acc[k]);
        }

        const auto& fromPsqt = computed.psqtAccumulation[c];
        auto& midPsqt = middle_state.psqtAccumulation[c];
        auto& toPsqt  = target_state.psqtAccumulation[c];
        psqt_vec_t psqt[Tiling::NumPsqtRegs];
        for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
        {
            const usize off = j * Tiling::PsqtTileHeight;
            auto* fromTile = reinterpret_cast<const psqt_vec_t*>(&fromPsqt[off]);
            auto* midTile  = reinterpret_cast<psqt_vec_t*>(&midPsqt[off]);
            auto* toTile   = reinterpret_cast<psqt_vec_t*>(&toPsqt[off]);
            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k) psqt[k] = fromTile[k];

            apply_psqt<-1>(j, psqt, psq_removed[0][c], featureTransformer.psqtWeights.data());
            apply_psqt<+1>(j, psqt, psq_added[0][c], featureTransformer.psqtWeights.data());
            apply_psqt<-1>(j, psqt, thr_removed[0][c], featureTransformer.threatAndPpPsqtWeights.data());
            apply_psqt<+1>(j, psqt, thr_added[0][c], featureTransformer.threatAndPpPsqtWeights.data());
            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k) vec_store_psqt(&midTile[k], psqt[k]);

            apply_psqt<-1>(j, psqt, psq_removed[1][c], featureTransformer.psqtWeights.data());
            apply_psqt<+1>(j, psqt, psq_added[1][c], featureTransformer.psqtWeights.data());
            apply_psqt<-1>(j, psqt, thr_removed[1][c], featureTransformer.threatAndPpPsqtWeights.data());
            apply_psqt<+1>(j, psqt, thr_added[1][c], featureTransformer.threatAndPpPsqtWeights.data());
            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k) vec_store_psqt(&toTile[k], psqt[k]);
        }
    }

    middle_state.computed[WHITE] = middle_state.computed[BLACK] = true;
    target_state.computed[WHITE] = target_state.computed[BLACK] = true;
#endif
}

'''
if s.count(anchor) != 1:
    raise SystemExit('helper insertion anchor')
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s)
