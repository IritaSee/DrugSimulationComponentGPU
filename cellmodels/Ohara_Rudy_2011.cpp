/*
   There are a total of 198 entries in the algebraic variable array.
   There are a total of 41 entries in each of the rate and state variable arrays.
   There are a total of 139+2 entries in the constant variable array.
 */

#include "Ohara_Rudy_2011.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "../modules/glob_funct.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * TIME is time in component environment (millisecond).
 * CONSTANTS[celltype] is celltype in component environment (dimensionless).
 * CONSTANTS[nao] is nao in component extracellular (millimolar).
 * CONSTANTS[cao] is cao in component extracellular (millimolar).
 * CONSTANTS[ko] is ko in component extracellular (millimolar).
 * CONSTANTS[R] is R in component physical_constants (joule_per_kilomole_kelvin).
 * CONSTANTS[T] is T in component physical_constants (kelvin).
 * CONSTANTS[F] is F in component physical_constants (coulomb_per_mole).
 * CONSTANTS[zna] is zna in component physical_constants (dimensionless).
 * CONSTANTS[zca] is zca in component physical_constants (dimensionless).
 * CONSTANTS[zk] is zk in component physical_constants (dimensionless).
 * CONSTANTS[L] is L in component cell_geometry (centimeter).
 * CONSTANTS[rad] is rad in component cell_geometry (centimeter).
 * CONSTANTS[vcell] is vcell in component cell_geometry (microliter).
 * CONSTANTS[Ageo] is Ageo in component cell_geometry (centimeter_squared).
 * CONSTANTS[Acap] is Acap in component cell_geometry (centimeter_squared).
 * CONSTANTS[vmyo] is vmyo in component cell_geometry (microliter).
 * CONSTANTS[vnsr] is vnsr in component cell_geometry (microliter).
 * CONSTANTS[vjsr] is vjsr in component cell_geometry (microliter).
 * CONSTANTS[vss] is vss in component cell_geometry (microliter).
 * STATES[V] is v in component membrane (millivolt).
 * ALGEBRAIC[vffrt] is vffrt in component membrane (coulomb_per_mole).
 * ALGEBRAIC[vfrt] is vfrt in component membrane (dimensionless).
 * ALGEBRAIC[INa] is INa in component INa (microA_per_microF).
 * ALGEBRAIC[INaL] is INaL in component INaL (microA_per_microF).
 * ALGEBRAIC[Ito] is Ito in component Ito (microA_per_microF).
 * ALGEBRAIC[ICaL] is ICaL in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa] is ICaNa in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK] is ICaK in component ICaL (microA_per_microF).
 * ALGEBRAIC[IKr] is IKr in component IKr (microA_per_microF).
 * ALGEBRAIC[IKs] is IKs in component IKs (microA_per_microF).
 * ALGEBRAIC[IK1] is IK1 in component IK1 (microA_per_microF).
 * ALGEBRAIC[INaCa_i] is INaCa_i in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaCa_ss] is INaCa_ss in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaK] is INaK in component INaK (microA_per_microF).
 * ALGEBRAIC[INab] is INab in component INab (microA_per_microF).
 * ALGEBRAIC[IKb] is IKb in component IKb (microA_per_microF).
 * ALGEBRAIC[IpCa] is IpCa in component IpCa (microA_per_microF).
 * ALGEBRAIC[ICab] is ICab in component ICab (microA_per_microF).
 * ALGEBRAIC[Istim] is Istim in component membrane (microA_per_microF).
 * CONSTANTS[amp] is amp in component membrane (microA_per_microF).
 * CONSTANTS[duration] is duration in component membrane (millisecond).
 * CONSTANTS[KmCaMK] is KmCaMK in component CaMK (millimolar).
 * CONSTANTS[aCaMK] is aCaMK in component CaMK (per_millimolar_per_millisecond).
 * CONSTANTS[bCaMK] is bCaMK in component CaMK (per_millisecond).
 * CONSTANTS[CaMKo] is CaMKo in component CaMK (dimensionless).
 * CONSTANTS[KmCaM] is KmCaM in component CaMK (millimolar).
 * ALGEBRAIC[CaMKb] is CaMKb in component CaMK (millimolar).
 * ALGEBRAIC[CaMKa] is CaMKa in component CaMK (millimolar).
 * STATES[CaMKt] is CaMKt in component CaMK (millimolar).
 * STATES[cass] is cass in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax_b] is cmdnmax_b in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax] is cmdnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcmdn] is kmcmdn in component intracellular_ions (millimolar).
 * CONSTANTS[trpnmax] is trpnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmtrpn] is kmtrpn in component intracellular_ions (millimolar).
 * CONSTANTS[BSRmax] is BSRmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSR] is KmBSR in component intracellular_ions (millimolar).
 * CONSTANTS[BSLmax] is BSLmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSL] is KmBSL in component intracellular_ions (millimolar).
 * CONSTANTS[csqnmax] is csqnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcsqn] is kmcsqn in component intracellular_ions (millimolar).
 * STATES[nai] is nai in component intracellular_ions (millimolar).
 * STATES[nass] is nass in component intracellular_ions (millimolar).
 * STATES[ki] is ki in component intracellular_ions (millimolar).
 * STATES[kss] is kss in component intracellular_ions (millimolar).
 * STATES[cansr] is cansr in component intracellular_ions (millimolar).
 * STATES[cajsr] is cajsr in component intracellular_ions (millimolar).
 * STATES[cai] is cai in component intracellular_ions (millimolar).
 * ALGEBRAIC[JdiffNa] is JdiffNa in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jdiff] is Jdiff in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jup] is Jup in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[JdiffK] is JdiffK in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel] is Jrel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jtr] is Jtr in component trans_flux (millimolar_per_millisecond).
 * ALGEBRAIC[Bcai] is Bcai in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcajsr] is Bcajsr in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcass] is Bcass in component intracellular_ions (dimensionless).
 * CONSTANTS[cm] is cm in component intracellular_ions (microF_per_centimeter_squared).
 * CONSTANTS[PKNa] is PKNa in component reversal_potentials (dimensionless).
 * ALGEBRAIC[ENa] is ENa in component reversal_potentials (millivolt).
 * ALGEBRAIC[EK] is EK in component reversal_potentials (millivolt).
 * ALGEBRAIC[EKs] is EKs in component reversal_potentials (millivolt).
 * ALGEBRAIC[mss] is mss in component INa (dimensionless).
 * ALGEBRAIC[tm] is tm in component INa (millisecond).
 * CONSTANTS[mssV1] is mssV1 in component INa (millivolt).
 * CONSTANTS[mssV2] is mssV2 in component INa (millivolt).
 * CONSTANTS[mtV1] is mtV1 in component INa (millivolt).
 * CONSTANTS[mtV2] is mtV2 in component INa (millivolt).
 * CONSTANTS[mtD1] is mtD1 in component INa (dimensionless).
 * CONSTANTS[mtD2] is mtD2 in component INa (dimensionless).
 * CONSTANTS[mtV3] is mtV3 in component INa (millivolt).
 * CONSTANTS[mtV4] is mtV4 in component INa (millivolt).
 * STATES[m] is m in component INa (dimensionless).
 * ALGEBRAIC[hss] is hss in component INa (dimensionless).
 * ALGEBRAIC[thf] is thf in component INa (millisecond).
 * ALGEBRAIC[ths] is ths in component INa (millisecond).
 * CONSTANTS[hssV1] is hssV1 in component INa (millivolt).
 * CONSTANTS[hssV2] is hssV2 in component INa (millivolt).
 * CONSTANTS[Ahs] is Ahs in component INa (dimensionless).
 * CONSTANTS[Ahf] is Ahf in component INa (dimensionless).
 * STATES[hf] is hf in component INa (dimensionless).
 * STATES[hs] is hs in component INa (dimensionless).
 * ALGEBRAIC[h] is h in component INa (dimensionless).
 * CONSTANTS[GNa] is GNa in component INa (milliS_per_microF).
 * ALGEBRAIC[jss] is jss in component INa (dimensionless).
 * ALGEBRAIC[tj] is tj in component INa (millisecond).
 * STATES[j] is j in component INa (dimensionless).
 * ALGEBRAIC[hssp] is hssp in component INa (dimensionless).
 * ALGEBRAIC[thsp] is thsp in component INa (millisecond).
 * STATES[hsp] is hsp in component INa (dimensionless).
 * ALGEBRAIC[hp] is hp in component INa (dimensionless).
 * ALGEBRAIC[tjp] is tjp in component INa (millisecond).
 * STATES[jp] is jp in component INa (dimensionless).
 * ALGEBRAIC[fINap] is fINap in component INa (dimensionless).
 * ALGEBRAIC[mLss] is mLss in component INaL (dimensionless).
 * ALGEBRAIC[tmL] is tmL in component INaL (millisecond).
 * STATES[mL] is mL in component INaL (dimensionless).
 * CONSTANTS[thL] is thL in component INaL (millisecond).
 * ALGEBRAIC[hLss] is hLss in component INaL (dimensionless).
 * STATES[hL] is hL in component INaL (dimensionless).
 * ALGEBRAIC[hLssp] is hLssp in component INaL (dimensionless).
 * CONSTANTS[thLp] is thLp in component INaL (millisecond).
 * STATES[hLp] is hLp in component INaL (dimensionless).
 * CONSTANTS[GNaL_b] is GNaL_b in component INaL (milliS_per_microF).
 * CONSTANTS[GNaL] is GNaL in component INaL (milliS_per_microF).
 * ALGEBRAIC[fINaLp] is fINaLp in component INaL (dimensionless).
 * CONSTANTS[Gto_b] is Gto_b in component Ito (milliS_per_microF).
 * ALGEBRAIC[ass] is ass in component Ito (dimensionless).
 * ALGEBRAIC[ta] is ta in component Ito (millisecond).
 * STATES[a] is a in component Ito (dimensionless).
 * ALGEBRAIC[iss] is iss in component Ito (dimensionless).
 * ALGEBRAIC[delta_epi] is delta_epi in component Ito (dimensionless).
 * ALGEBRAIC[tiF_b] is tiF_b in component Ito (millisecond).
 * ALGEBRAIC[tiS_b] is tiS_b in component Ito (millisecond).
 * ALGEBRAIC[tiF] is tiF in component Ito (millisecond).
 * ALGEBRAIC[tiS] is tiS in component Ito (millisecond).
 * ALGEBRAIC[AiF] is AiF in component Ito (dimensionless).
 * ALGEBRAIC[AiS] is AiS in component Ito (dimensionless).
 * STATES[iF] is iF in component Ito (dimensionless).
 * STATES[iS] is iS in component Ito (dimensionless).
 * ALGEBRAIC[i] is i in component Ito (dimensionless).
 * ALGEBRAIC[assp] is assp in component Ito (dimensionless).
 * STATES[ap] is ap in component Ito (dimensionless).
 * ALGEBRAIC[dti_develop] is dti_develop in component Ito (dimensionless).
 * ALGEBRAIC[dti_recover] is dti_recover in component Ito (dimensionless).
 * ALGEBRAIC[tiFp] is tiFp in component Ito (millisecond).
 * ALGEBRAIC[tiSp] is tiSp in component Ito (millisecond).
 * STATES[iFp] is iFp in component Ito (dimensionless).
 * STATES[iSp] is iSp in component Ito (dimensionless).
 * ALGEBRAIC[ip] is ip in component Ito (dimensionless).
 * CONSTANTS[Gto] is Gto in component Ito (milliS_per_microF).
 * ALGEBRAIC[fItop] is fItop in component Ito (dimensionless).
 * CONSTANTS[Kmn] is Kmn in component ICaL (millimolar).
 * CONSTANTS[k2n] is k2n in component ICaL (per_millisecond).
 * CONSTANTS[PCa_b] is PCa_b in component ICaL (dimensionless).
 * ALGEBRAIC[dss] is dss in component ICaL (dimensionless).
 * STATES[d] is d in component ICaL (dimensionless).
 * ALGEBRAIC[fss] is fss in component ICaL (dimensionless).
 * CONSTANTS[Aff] is Aff in component ICaL (dimensionless).
 * CONSTANTS[Afs] is Afs in component ICaL (dimensionless).
 * STATES[ff] is ff in component ICaL (dimensionless).
 * STATES[fs] is fs in component ICaL (dimensionless).
 * ALGEBRAIC[f] is f in component ICaL (dimensionless).
 * ALGEBRAIC[fcass] is fcass in component ICaL (dimensionless).
 * ALGEBRAIC[Afcaf] is Afcaf in component ICaL (dimensionless).
 * ALGEBRAIC[Afcas] is Afcas in component ICaL (dimensionless).
 * STATES[fcaf] is fcaf in component ICaL (dimensionless).
 * STATES[fcas] is fcas in component ICaL (dimensionless).
 * ALGEBRAIC[fca] is fca in component ICaL (dimensionless).
 * STATES[jca] is jca in component ICaL (dimensionless).
 * STATES[ffp] is ffp in component ICaL (dimensionless).
 * ALGEBRAIC[fp] is fp in component ICaL (dimensionless).
 * STATES[fcafp] is fcafp in component ICaL (dimensionless).
 * ALGEBRAIC[fcap] is fcap in component ICaL (dimensionless).
 * ALGEBRAIC[km2n] is km2n in component ICaL (per_millisecond).
 * ALGEBRAIC[anca] is anca in component ICaL (dimensionless).
 * STATES[nca] is nca in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL] is PhiCaL in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa] is PhiCaNa in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK] is PhiCaK in component ICaL (dimensionless).
 * CONSTANTS[PCa] is PCa in component ICaL (dimensionless).
 * CONSTANTS[PCap] is PCap in component ICaL (dimensionless).
 * CONSTANTS[PCaNa] is PCaNa in component ICaL (dimensionless).
 * CONSTANTS[PCaK] is PCaK in component ICaL (dimensionless).
 * CONSTANTS[PCaNap] is PCaNap in component ICaL (dimensionless).
 * CONSTANTS[PCaKp] is PCaKp in component ICaL (dimensionless).
 * ALGEBRAIC[fICaLp] is fICaLp in component ICaL (dimensionless).
 * ALGEBRAIC[td] is td in component ICaL (millisecond).
 * ALGEBRAIC[tff] is tff in component ICaL (millisecond).
 * ALGEBRAIC[tfs] is tfs in component ICaL (millisecond).
 * ALGEBRAIC[tfcaf] is tfcaf in component ICaL (millisecond).
 * ALGEBRAIC[tfcas] is tfcas in component ICaL (millisecond).
 * CONSTANTS[tjca] is tjca in component ICaL (millisecond).
 * ALGEBRAIC[tffp] is tffp in component ICaL (millisecond).
 * ALGEBRAIC[tfcafp] is tfcafp in component ICaL (millisecond).
 * CONSTANTS[GKr_b] is GKr_b in component IKr (milliS_per_microF).
 * CONSTANTS[GKr] is GKr in component IKr (milliS_per_microF).
 * ALGEBRAIC[xrss] is xrss in component IKr (dimensionless).
 * ALGEBRAIC[txrf] is txrf in component IKr (millisecond).
 * ALGEBRAIC[txrs] is txrs in component IKr (millisecond).
 * ALGEBRAIC[Axrf] is Axrf in component IKr (dimensionless).
 * ALGEBRAIC[Axrs] is Axrs in component IKr (dimensionless).
 * STATES[xrf] is xrf in component IKr (dimensionless).
 * STATES[xrs] is xrs in component IKr (dimensionless).
 * ALGEBRAIC[xr] is xr in component IKr (dimensionless).
 * ALGEBRAIC[rkr] is rkr in component IKr (dimensionless).
 * CONSTANTS[GKs_b] is GKs_b in component IKs (milliS_per_microF).
 * CONSTANTS[GKs] is GKs in component IKs (milliS_per_microF).
 * ALGEBRAIC[xs1ss] is xs1ss in component IKs (dimensionless).
 * ALGEBRAIC[xs2ss] is xs2ss in component IKs (dimensionless).
 * ALGEBRAIC[txs1] is txs1 in component IKs (millisecond).
 * STATES[xs1] is xs1 in component IKs (dimensionless).
 * STATES[xs2] is xs2 in component IKs (dimensionless).
 * ALGEBRAIC[KsCa] is KsCa in component IKs (dimensionless).
 * ALGEBRAIC[txs2] is txs2 in component IKs (millisecond).
 * CONSTANTS[GK1] is GK1 in component IK1 (milliS_per_microF).
 * CONSTANTS[GK1_b] is GK1_b in component IK1 (milliS_per_microF).
 * ALGEBRAIC[xk1ss] is xk1ss in component IK1 (dimensionless).
 * ALGEBRAIC[txk1] is txk1 in component IK1 (millisecond).
 * STATES[xk1] is xk1 in component IK1 (dimensionless).
 * ALGEBRAIC[rk1] is rk1 in component IK1 (millisecond).
 * CONSTANTS[kna1] is kna1 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna2] is kna2 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna3] is kna3 in component INaCa_i (per_millisecond).
 * CONSTANTS[kasymm] is kasymm in component INaCa_i (dimensionless).
 * CONSTANTS[wna] is wna in component INaCa_i (dimensionless).
 * CONSTANTS[wca] is wca in component INaCa_i (dimensionless).
 * CONSTANTS[wnaca] is wnaca in component INaCa_i (dimensionless).
 * CONSTANTS[kcaon] is kcaon in component INaCa_i (per_millisecond).
 * CONSTANTS[kcaoff] is kcaoff in component INaCa_i (per_millisecond).
 * CONSTANTS[qna] is qna in component INaCa_i (dimensionless).
 * CONSTANTS[qca] is qca in component INaCa_i (dimensionless).
 * ALGEBRAIC[hna] is hna in component INaCa_i (dimensionless).
 * ALGEBRAIC[hca] is hca in component INaCa_i (dimensionless).
 * CONSTANTS[KmCaAct] is KmCaAct in component INaCa_i (millimolar).
 * CONSTANTS[Gncx_b] is Gncx_b in component INaCa_i (milliS_per_microF).
 * CONSTANTS[Gncx] is Gncx in component INaCa_i (milliS_per_microF).
 * ALGEBRAIC[h1_i] is h1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_i] is h2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_i] is h3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_i] is h4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_i] is h5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_i] is h6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_i] is h7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_i] is h8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_i] is h9_i in component INaCa_i (dimensionless).
 * CONSTANTS[h10_i] is h10_i in component INaCa_i (dimensionless).
 * CONSTANTS[h11_i] is h11_i in component INaCa_i (dimensionless).
 * CONSTANTS[h12_i] is h12_i in component INaCa_i (dimensionless).
 * CONSTANTS[k1_i] is k1_i in component INaCa_i (dimensionless).
 * CONSTANTS[k2_i] is k2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_i] is k3p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_i] is k3pp_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_i] is k3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_i] is k4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_i] is k4p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_i] is k4pp_i in component INaCa_i (dimensionless).
 * CONSTANTS[k5_i] is k5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_i] is k6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_i] is k7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_i] is k8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_i] is x1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_i] is x2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_i] is x3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_i] is x4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_i] is E1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_i] is E2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_i] is E3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_i] is E4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_i] is allo_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_i] is JncxNa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_i] is JncxCa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[h1_ss] is h1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_ss] is h2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_ss] is h3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_ss] is h4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_ss] is h5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_ss] is h6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_ss] is h7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_ss] is h8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_ss] is h9_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h10_ss] is h10_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h11_ss] is h11_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h12_ss] is h12_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k1_ss] is k1_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k2_ss] is k2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_ss] is k3p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_ss] is k3pp_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_ss] is k3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_ss] is k4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_ss] is k4p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_ss] is k4pp_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k5_ss] is k5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_ss] is k6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_ss] is k7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_ss] is k8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_ss] is x1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_ss] is x2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_ss] is x3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_ss] is x4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_ss] is E1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_ss] is E2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_ss] is E3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_ss] is E4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_ss] is allo_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_ss] is JncxNa_ss in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_ss] is JncxCa_ss in component INaCa_i (millimolar_per_millisecond).
 * CONSTANTS[k1p] is k1p in component INaK (per_millisecond).
 * CONSTANTS[k1m] is k1m in component INaK (per_millisecond).
 * CONSTANTS[k2p] is k2p in component INaK (per_millisecond).
 * CONSTANTS[k2m] is k2m in component INaK (per_millisecond).
 * CONSTANTS[k3p] is k3p in component INaK (per_millisecond).
 * CONSTANTS[k3m] is k3m in component INaK (per_millisecond).
 * CONSTANTS[k4p] is k4p in component INaK (per_millisecond).
 * CONSTANTS[k4m] is k4m in component INaK (per_millisecond).
 * CONSTANTS[Knai0] is Knai0 in component INaK (millimolar).
 * CONSTANTS[Knao0] is Knao0 in component INaK (millimolar).
 * CONSTANTS[delta] is delta in component INaK (millivolt).
 * CONSTANTS[Kki] is Kki in component INaK (per_millisecond).
 * CONSTANTS[Kko] is Kko in component INaK (per_millisecond).
 * CONSTANTS[MgADP] is MgADP in component INaK (millimolar).
 * CONSTANTS[MgATP] is MgATP in component INaK (millimolar).
 * CONSTANTS[Kmgatp] is Kmgatp in component INaK (millimolar).
 * CONSTANTS[H] is H in component INaK (millimolar).
 * CONSTANTS[eP] is eP in component INaK (dimensionless).
 * CONSTANTS[Khp] is Khp in component INaK (millimolar).
 * CONSTANTS[Knap] is Knap in component INaK (millimolar).
 * CONSTANTS[Kxkur] is Kxkur in component INaK (millimolar).
 * CONSTANTS[Pnak_b] is Pnak_b in component INaK (milliS_per_microF).
 * CONSTANTS[Pnak] is Pnak in component INaK (milliS_per_microF).
 * ALGEBRAIC[Knai] is Knai in component INaK (millimolar).
 * ALGEBRAIC[Knao] is Knao in component INaK (millimolar).
 * ALGEBRAIC[P] is P in component INaK (dimensionless).
 * ALGEBRAIC[a1] is a1 in component INaK (dimensionless).
 * CONSTANTS[b1] is b1 in component INaK (dimensionless).
 * CONSTANTS[a2] is a2 in component INaK (dimensionless).
 * ALGEBRAIC[b2] is b2 in component INaK (dimensionless).
 * ALGEBRAIC[a3] is a3 in component INaK (dimensionless).
 * ALGEBRAIC[b3] is b3 in component INaK (dimensionless).
 * CONSTANTS[a4] is a4 in component INaK (dimensionless).
 * ALGEBRAIC[b4] is b4 in component INaK (dimensionless).
 * ALGEBRAIC[x1] is x1 in component INaK (dimensionless).
 * ALGEBRAIC[x2] is x2 in component INaK (dimensionless).
 * ALGEBRAIC[x3] is x3 in component INaK (dimensionless).
 * ALGEBRAIC[x4] is x4 in component INaK (dimensionless).
 * ALGEBRAIC[E1] is E1 in component INaK (dimensionless).
 * ALGEBRAIC[E2] is E2 in component INaK (dimensionless).
 * ALGEBRAIC[E3] is E3 in component INaK (dimensionless).
 * ALGEBRAIC[E4] is E4 in component INaK (dimensionless).
 * ALGEBRAIC[JnakNa] is JnakNa in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[JnakK] is JnakK in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[xkb] is xkb in component IKb (dimensionless).
 * CONSTANTS[GKb_b] is GKb_b in component IKb (milliS_per_microF).
 * CONSTANTS[GKb] is GKb in component IKb (milliS_per_microF).
 * CONSTANTS[PNab] is PNab in component INab (milliS_per_microF).
 * CONSTANTS[PCab] is PCab in component ICab (milliS_per_microF).
 * CONSTANTS[GpCa] is GpCa in component IpCa (milliS_per_microF).
 * CONSTANTS[KmCap] is KmCap in component IpCa (millimolar).
 * CONSTANTS[bt] is bt in component ryr (millisecond).
 * CONSTANTS[a_rel] is a_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf] is Jrel_inf in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel] is tau_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_infp] is Jrel_infp in component ryr (dimensionless).
 * ALGEBRAIC[Jrel_temp] is Jrel_temp in component ryr (dimensionless).
 * ALGEBRAIC[tau_relp] is tau_relp in component ryr (millisecond).
 * STATES[Jrelnp] is Jrelnp in component ryr (dimensionless).
 * STATES[Jrelp] is Jrelp in component ryr (dimensionless).
 * CONSTANTS[btp] is btp in component ryr (millisecond).
 * CONSTANTS[a_relp] is a_relp in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf_temp] is Jrel_inf_temp in component ryr (dimensionless).
 * ALGEBRAIC[fJrelp] is fJrelp in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel_temp] is tau_rel_temp in component ryr (millisecond).
 * ALGEBRAIC[tau_relp_temp] is tau_relp_temp in component ryr (millisecond).
 * CONSTANTS[upScale] is upScale in component SERCA (dimensionless).
 * ALGEBRAIC[Jupnp] is Jupnp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[Jupp] is Jupp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[fJupp] is fJupp in component SERCA (dimensionless).
 * ALGEBRAIC[Jleak] is Jleak in component SERCA (millimolar_per_millisecond).
 * RATES[V] is d/dt v in component membrane (millivolt).
 * RATES[CaMKt] is d/dt CaMKt in component CaMK (millimolar).
 * RATES[nai] is d/dt nai in component intracellular_ions (millimolar).
 * RATES[nass] is d/dt nass in component intracellular_ions (millimolar).
 * RATES[ki] is d/dt ki in component intracellular_ions (millimolar).
 * RATES[kss] is d/dt kss in component intracellular_ions (millimolar).
 * RATES[cai] is d/dt cai in component intracellular_ions (millimolar).
 * RATES[cass] is d/dt cass in component intracellular_ions (millimolar).
 * RATES[cansr] is d/dt cansr in component intracellular_ions (millimolar).
 * RATES[cajsr] is d/dt cajsr in component intracellular_ions (millimolar).
 * RATES[m] is d/dt m in component INa (dimensionless).
 * RATES[hf] is d/dt hf in component INa (dimensionless).
 * RATES[hs] is d/dt hs in component INa (dimensionless).
 * RATES[j] is d/dt j in component INa (dimensionless).
 * RATES[hsp] is d/dt hsp in component INa (dimensionless).
 * RATES[jp] is d/dt jp in component INa (dimensionless).
 * RATES[mL] is d/dt mL in component INaL (dimensionless).
 * RATES[hL] is d/dt hL in component INaL (dimensionless).
 * RATES[hLp] is d/dt hLp in component INaL (dimensionless).
 * RATES[a] is d/dt a in component Ito (dimensionless).
 * RATES[iF] is d/dt iF in component Ito (dimensionless).
 * RATES[iS] is d/dt iS in component Ito (dimensionless).
 * RATES[ap] is d/dt ap in component Ito (dimensionless).
 * RATES[iFp] is d/dt iFp in component Ito (dimensionless).
 * RATES[iSp] is d/dt iSp in component Ito (dimensionless).
 * RATES[d] is d/dt d in component ICaL (dimensionless).
 * RATES[ff] is d/dt ff in component ICaL (dimensionless).
 * RATES[fs] is d/dt fs in component ICaL (dimensionless).
 * RATES[fcaf] is d/dt fcaf in component ICaL (dimensionless).
 * RATES[fcas] is d/dt fcas in component ICaL (dimensionless).
 * RATES[jca] is d/dt jca in component ICaL (dimensionless).
 * RATES[ffp] is d/dt ffp in component ICaL (dimensionless).
 * RATES[fcafp] is d/dt fcafp in component ICaL (dimensionless).
 * RATES[nca] is d/dt nca in component ICaL (dimensionless).
 * RATES[xrf] is d/dt xrf in component IKr (dimensionless).
 * RATES[xrs] is d/dt xrs in component IKr (dimensionless).
 * RATES[xs1] is d/dt xs1 in component IKs (dimensionless).
 * RATES[xs2] is d/dt xs2 in component IKs (dimensionless).
 * RATES[xk1] is d/dt xk1 in component IK1 (dimensionless).
 * RATES[Jrelnp] is d/dt Jrelnp in component ryr (dimensionless).
 * RATES[Jrelp] is d/dt Jrelp in component ryr (dimensionless).
 */

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, int offset)
{

int num_of_constants = 146;
int num_of_states = 41;
// printf("%d\n", offset);
CONSTANTS[(offset * num_of_constants) + nao] = 140;
CONSTANTS[(offset * num_of_constants) + cao] = 1.8;
CONSTANTS[(offset * num_of_constants) + ko] = 5.4;
CONSTANTS[(offset * num_of_constants) + R] = 8314;
CONSTANTS[(offset * num_of_constants) + T] = 310;
CONSTANTS[(offset * num_of_constants) + F] = 96485;
CONSTANTS[(offset * num_of_constants) + zna] = 1;
CONSTANTS[(offset * num_of_constants) + zca] = 2;
CONSTANTS[(offset * num_of_constants) + zk] = 1;
CONSTANTS[(offset * num_of_constants) + L] = 0.01;
CONSTANTS[(offset * num_of_constants) + rad] = 0.0011;
CONSTANTS[(offset * num_of_constants) + stim_start] = 10.0;
CONSTANTS[(offset * num_of_constants) + BCL] = 1000.0;
STATES[(offset * num_of_states) + V] = -87;
CONSTANTS[(offset * num_of_constants) + amp] = -80;
CONSTANTS[(offset * num_of_constants) + duration] = 0.5;
CONSTANTS[(offset * num_of_constants) + KmCaMK] = 0.15;
CONSTANTS[(offset * num_of_constants) + aCaMK] = 0.05;
CONSTANTS[(offset * num_of_constants) + bCaMK] = 0.00068;
CONSTANTS[(offset * num_of_constants) + CaMKo] = 0.05;
CONSTANTS[(offset * num_of_constants) + KmCaM] = 0.0015;
STATES[(offset * num_of_states) + CaMKt] = 0;
STATES[(offset * num_of_states) + cass] = 1e-4;
CONSTANTS[(offset * num_of_constants) + cmdnmax_b] = 0.05;
CONSTANTS[(offset * num_of_constants) + kmcmdn] = 0.00238;
CONSTANTS[(offset * num_of_constants) + trpnmax] = 0.07;
CONSTANTS[(offset * num_of_constants) + kmtrpn] = 0.0005;
CONSTANTS[(offset * num_of_constants) + BSRmax] = 0.047;
CONSTANTS[(offset * num_of_constants) + KmBSR] = 0.00087;
CONSTANTS[(offset * num_of_constants) + BSLmax] = 1.124;
CONSTANTS[(offset * num_of_constants) + KmBSL] = 0.0087;
CONSTANTS[(offset * num_of_constants) + csqnmax] = 10;
CONSTANTS[(offset * num_of_constants) + kmcsqn] = 0.8;
STATES[(offset * num_of_states) + nai] = 7;
STATES[(offset * num_of_states) + nass] = 7;
STATES[(offset * num_of_states) + ki] = 145;
STATES[(offset * num_of_states) + kss] = 145;
STATES[(offset * num_of_states) + cansr] = 1.2;
STATES[(offset * num_of_states) + cajsr] = 1.2;
STATES[(offset * num_of_states) + cai] = 1e-4;
CONSTANTS[(offset * num_of_constants) + cm] = 1;
CONSTANTS[(offset * num_of_constants) + PKNa] = 0.01833;
CONSTANTS[(offset * num_of_constants) + mssV1] = 39.57;
CONSTANTS[(offset * num_of_constants) + mssV2] = 9.871;
CONSTANTS[(offset * num_of_constants) + mtV1] = 11.64;
CONSTANTS[(offset * num_of_constants) + mtV2] = 34.77;
CONSTANTS[(offset * num_of_constants) + mtD1] = 6.765;
CONSTANTS[(offset * num_of_constants) + mtD2] = 8.552;
CONSTANTS[(offset * num_of_constants) + mtV3] = 77.42;
CONSTANTS[(offset * num_of_constants) + mtV4] = 5.955;
STATES[(offset * num_of_states) + m] = 0;
CONSTANTS[(offset * num_of_constants) + hssV1] = 82.9;
CONSTANTS[(offset * num_of_constants) + hssV2] = 6.086;
CONSTANTS[(offset * num_of_constants) + Ahf] = 0.99;
STATES[(offset * num_of_states) + hf] = 1;
STATES[(offset * num_of_states) + hs] = 1;
CONSTANTS[(offset * num_of_constants) + GNa] = 75;
STATES[(offset * num_of_states) + j] = 1;
STATES[(offset * num_of_states) + hsp] = 1;
STATES[(offset * num_of_states) + jp] = 1;
STATES[(offset * num_of_states) + mL] = 0;
CONSTANTS[(offset * num_of_constants) + thL] = 200;
STATES[(offset * num_of_states) + hL] = 1;
STATES[(offset * num_of_states) + hLp] = 1;
CONSTANTS[(offset * num_of_constants) + GNaL_b] = 0.0075;
CONSTANTS[(offset * num_of_constants) + Gto_b] = 0.02;
STATES[(offset * num_of_states) + a] = 0;
STATES[(offset * num_of_states) + iF] = 1;
STATES[(offset * num_of_states) + iS] = 1;
STATES[(offset * num_of_states) + ap] = 0;
STATES[(offset * num_of_states) + iFp] = 1;
STATES[(offset * num_of_states) + iSp] = 1;
CONSTANTS[(offset * num_of_constants) + Kmn] = 0.002;
CONSTANTS[(offset * num_of_constants) + k2n] = 1000;
CONSTANTS[(offset * num_of_constants) + PCa_b] = 0.0001;
STATES[(offset * num_of_states) + d] = 0;
STATES[(offset * num_of_states) + ff] = 1;
STATES[(offset * num_of_states) + fs] = 1;
STATES[(offset * num_of_states) + fcaf] = 1;
STATES[(offset * num_of_states) + fcas] = 1;
STATES[(offset * num_of_states) + jca] = 1;
STATES[(offset * num_of_states) + ffp] = 1;
STATES[(offset * num_of_states) + fcafp] = 1;
STATES[(offset * num_of_states) + nca] = 0;
CONSTANTS[(offset * num_of_constants) + GKr_b] = 0.046;
STATES[(offset * num_of_states) + xrf] = 0;
STATES[(offset * num_of_states) + xrs] = 0;
CONSTANTS[(offset * num_of_constants) + GKs_b] = 0.0034;
STATES[(offset * num_of_states) + xs1] = 0;
STATES[(offset * num_of_states) + xs2] = 0;
CONSTANTS[(offset * num_of_constants) + GK1_b] = 0.1908;
STATES[(offset * num_of_states) + xk1] = 1;
CONSTANTS[(offset * num_of_constants) + kna1] = 15;
CONSTANTS[(offset * num_of_constants) + kna2] = 5;
CONSTANTS[(offset * num_of_constants) + kna3] = 88.12;
CONSTANTS[(offset * num_of_constants) + kasymm] = 12.5;
CONSTANTS[(offset * num_of_constants) + wna] = 6e4;
CONSTANTS[(offset * num_of_constants) + wca] = 6e4;
CONSTANTS[(offset * num_of_constants) + wnaca] = 5e3;
CONSTANTS[(offset * num_of_constants) + kcaon] = 1.5e6;
CONSTANTS[(offset * num_of_constants) + kcaoff] = 5e3;
CONSTANTS[(offset * num_of_constants) + qna] = 0.5224;
CONSTANTS[(offset * num_of_constants) + qca] = 0.167;
CONSTANTS[(offset * num_of_constants) + KmCaAct] = 150e-6;
CONSTANTS[(offset * num_of_constants) + Gncx_b] = 0.0008;
CONSTANTS[(offset * num_of_constants) + k1p] = 949.5;
CONSTANTS[(offset * num_of_constants) + k1m] = 182.4;
CONSTANTS[(offset * num_of_constants) + k2p] = 687.2;
CONSTANTS[(offset * num_of_constants) + k2m] = 39.4;
CONSTANTS[(offset * num_of_constants) + k3p] = 1899;
CONSTANTS[(offset * num_of_constants) + k3m] = 79300;
CONSTANTS[(offset * num_of_constants) + k4p] = 639;
CONSTANTS[(offset * num_of_constants) + k4m] = 40;
CONSTANTS[(offset * num_of_constants) + Knai0] = 9.073;
CONSTANTS[(offset * num_of_constants) + Knao0] = 27.78;
CONSTANTS[(offset * num_of_constants) + delta] = -0.155;
CONSTANTS[(offset * num_of_constants) + Kki] = 0.5;
CONSTANTS[(offset * num_of_constants) + Kko] = 0.3582;
CONSTANTS[(offset * num_of_constants) + MgADP] = 0.05;
CONSTANTS[(offset * num_of_constants) + MgATP] = 9.8;
CONSTANTS[(offset * num_of_constants) + Kmgatp] = 1.698e-7;
CONSTANTS[(offset * num_of_constants) + H] = 1e-7;
CONSTANTS[(offset * num_of_constants) + eP] = 4.2;
CONSTANTS[(offset * num_of_constants) + Khp] = 1.698e-7;
CONSTANTS[(offset * num_of_constants) + Knap] = 224;
CONSTANTS[(offset * num_of_constants) + Kxkur] = 292;
CONSTANTS[(offset * num_of_constants) + Pnak_b] = 30;
CONSTANTS[(offset * num_of_constants) + GKb_b] = 0.003;
CONSTANTS[(offset * num_of_constants) + PNab] = 3.75e-10;
CONSTANTS[(offset * num_of_constants) + PCab] = 2.5e-8;
CONSTANTS[(offset * num_of_constants) + GpCa] = 0.0005;
CONSTANTS[(offset * num_of_constants) + KmCap] = 0.0005;
CONSTANTS[(offset * num_of_constants) + bt] = 4.75;
STATES[(offset * num_of_states) + Jrelnp] = 0;
STATES[(offset * num_of_states) + Jrelp] = 0;
CONSTANTS[(offset * num_of_constants) + cmdnmax] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + cmdnmax_b]*1.30000 : CONSTANTS[(offset * num_of_constants) + cmdnmax_b]);
CONSTANTS[(offset * num_of_constants) + Ahs] = 1.00000 - CONSTANTS[(offset * num_of_constants) + Ahf];
CONSTANTS[(offset * num_of_constants) + thLp] = 3.00000 * CONSTANTS[(offset * num_of_constants) + thL];
CONSTANTS[(offset * num_of_constants) + GNaL] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + GNaL_b]*0.600000 : CONSTANTS[(offset * num_of_constants) + GNaL_b]);
CONSTANTS[(offset * num_of_constants) + Gto] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + Gto_b]*4.00000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + Gto_b]*4.00000 : CONSTANTS[(offset * num_of_constants) + Gto_b]);
CONSTANTS[(offset * num_of_constants) + Aff] = 0.600000;
CONSTANTS[(offset * num_of_constants) + PCa] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + PCa_b]*1.20000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + PCa_b]*2.50000 : CONSTANTS[(offset * num_of_constants) + PCa_b]);
CONSTANTS[(offset * num_of_constants) + tjca] = 75.0000;
CONSTANTS[(offset * num_of_constants) + GKr] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + GKr_b]*1.30000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + GKr_b]*0.800000 : CONSTANTS[(offset * num_of_constants) + GKr_b]);
CONSTANTS[(offset * num_of_constants) + GKs] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + GKs_b]*1.40000 : CONSTANTS[(offset * num_of_constants) + GKs_b]);
CONSTANTS[(offset * num_of_constants) + GK1] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + GK1_b]*1.20000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + GK1_b]*1.30000 : CONSTANTS[(offset * num_of_constants) + GK1_b]);
CONSTANTS[(offset * num_of_constants) + vcell] =  1000.00*3.14000*CONSTANTS[(offset * num_of_constants) + rad]*CONSTANTS[(offset * num_of_constants) + rad]*CONSTANTS[(offset * num_of_constants) + L];
CONSTANTS[(offset * num_of_constants) + GKb] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + GKb_b]*0.600000 : CONSTANTS[(offset * num_of_constants) + GKb_b]);
CONSTANTS[(offset * num_of_constants) + a_rel] =  0.500000*CONSTANTS[(offset * num_of_constants) + bt];
CONSTANTS[(offset * num_of_constants) + btp] =  1.25000*CONSTANTS[(offset * num_of_constants) + bt];
CONSTANTS[(offset * num_of_constants) + upScale] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.0000 ? 1.30000 : 1.00000);
CONSTANTS[(offset * num_of_constants) + Afs] = 1.00000 - CONSTANTS[(offset * num_of_constants) + Aff];
CONSTANTS[(offset * num_of_constants) + PCap] =  1.10000*CONSTANTS[(offset * num_of_constants) + PCa];
CONSTANTS[(offset * num_of_constants) + PCaNa] =  0.00125000*CONSTANTS[(offset * num_of_constants) + PCa];
CONSTANTS[(offset * num_of_constants) + PCaK] =  0.000357400*CONSTANTS[(offset * num_of_constants) + PCa];
CONSTANTS[(offset * num_of_constants) + Ageo] =  2.00000*3.14000*CONSTANTS[(offset * num_of_constants) + rad]*CONSTANTS[(offset * num_of_constants) + rad]+ 2.00000*3.14000*CONSTANTS[(offset * num_of_constants) + rad]*CONSTANTS[(offset * num_of_constants) + L];
CONSTANTS[(offset * num_of_constants) + a_relp] =  0.500000*CONSTANTS[(offset * num_of_constants) + btp];
CONSTANTS[(offset * num_of_constants) + PCaNap] =  0.00125000*CONSTANTS[(offset * num_of_constants) + PCap];
CONSTANTS[(offset * num_of_constants) + PCaKp] =  0.000357400*CONSTANTS[(offset * num_of_constants) + PCap];
CONSTANTS[(offset * num_of_constants) + Acap] =  2.00000*CONSTANTS[(offset * num_of_constants) + Ageo];
CONSTANTS[(offset * num_of_constants) + vmyo] =  0.680000*CONSTANTS[(offset * num_of_constants) + vcell];
CONSTANTS[(offset * num_of_constants) + vnsr] =  0.0552000*CONSTANTS[(offset * num_of_constants) + vcell];
CONSTANTS[(offset * num_of_constants) + vjsr] =  0.00480000*CONSTANTS[(offset * num_of_constants) + vcell];
CONSTANTS[(offset * num_of_constants) + vss] =  0.0200000*CONSTANTS[(offset * num_of_constants) + vcell];
CONSTANTS[(offset * num_of_constants) + h10_i] = CONSTANTS[(offset * num_of_constants) + kasymm]+1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna2]);
CONSTANTS[(offset * num_of_constants) + h11_i] = ( CONSTANTS[(offset * num_of_constants) + nao]*CONSTANTS[(offset * num_of_constants) + nao])/( CONSTANTS[(offset * num_of_constants) + h10_i]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
CONSTANTS[(offset * num_of_constants) + h12_i] = 1.00000/CONSTANTS[(offset * num_of_constants) + h10_i];
CONSTANTS[(offset * num_of_constants) + k1_i] =  CONSTANTS[(offset * num_of_constants) + h12_i]*CONSTANTS[(offset * num_of_constants) + cao]*CONSTANTS[(offset * num_of_constants) + kcaon];
CONSTANTS[(offset * num_of_constants) + k2_i] = CONSTANTS[(offset * num_of_constants) + kcaoff];
CONSTANTS[(offset * num_of_constants) + k5_i] = CONSTANTS[(offset * num_of_constants) + kcaoff];
CONSTANTS[(offset * num_of_constants) + Gncx] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + Gncx_b]*1.10000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + Gncx_b]*1.40000 : CONSTANTS[(offset * num_of_constants) + Gncx_b]);
CONSTANTS[(offset * num_of_constants) + h10_ss] = CONSTANTS[(offset * num_of_constants) + kasymm]+1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna2]);
CONSTANTS[(offset * num_of_constants) + h11_ss] = ( CONSTANTS[(offset * num_of_constants) + nao]*CONSTANTS[(offset * num_of_constants) + nao])/( CONSTANTS[(offset * num_of_constants) + h10_ss]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
CONSTANTS[(offset * num_of_constants) + h12_ss] = 1.00000/CONSTANTS[(offset * num_of_constants) + h10_ss];
CONSTANTS[(offset * num_of_constants) + k1_ss] =  CONSTANTS[(offset * num_of_constants) + h12_ss]*CONSTANTS[(offset * num_of_constants) + cao]*CONSTANTS[(offset * num_of_constants) + kcaon];
CONSTANTS[(offset * num_of_constants) + k2_ss] = CONSTANTS[(offset * num_of_constants) + kcaoff];
CONSTANTS[(offset * num_of_constants) + k5_ss] = CONSTANTS[(offset * num_of_constants) + kcaoff];
CONSTANTS[(offset * num_of_constants) + b1] =  CONSTANTS[(offset * num_of_constants) + k1m]*CONSTANTS[(offset * num_of_constants) + MgADP];
CONSTANTS[(offset * num_of_constants) + a2] = CONSTANTS[(offset * num_of_constants) + k2p];
CONSTANTS[(offset * num_of_constants) + a4] = (( CONSTANTS[(offset * num_of_constants) + k4p]*CONSTANTS[(offset * num_of_constants) + MgATP])/CONSTANTS[(offset * num_of_constants) + Kmgatp])/(1.00000+CONSTANTS[(offset * num_of_constants) + MgATP]/CONSTANTS[(offset * num_of_constants) + Kmgatp]);
CONSTANTS[(offset * num_of_constants) + Pnak] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ?  CONSTANTS[(offset * num_of_constants) + Pnak_b]*0.900000 : CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  CONSTANTS[(offset * num_of_constants) + Pnak_b]*0.700000 : CONSTANTS[(offset * num_of_constants) + Pnak_b]);
}

__device__ void ___applyDutta(double *CONSTANTS, int offset)
{
int num_of_constants = 146;
//sisanya ganti jadi G (GKs for example)
CONSTANTS[GKs + (offset * num_of_constants)] *= 1.870;
CONSTANTS[GKr + (offset * num_of_constants)] *= 1.013;
CONSTANTS[GK1 + (offset * num_of_constants)] *= 1.698;
CONSTANTS[PCa + (offset * num_of_constants)] *= 1.007; //pca
CONSTANTS[GNaL + (offset * num_of_constants)] *= 2.661;
}

/*==============*/
/* Added by ALI */
/*==============*/
__device__ void ___applyCvar(double *CONSTANTS, double *cvar, int offset)
{
  int num_of_constants = 146;

  CONSTANTS[(offset * num_of_constants) +GNa] *= cvar[0 + (offset*18)];		// GNa
  CONSTANTS[(offset * num_of_constants) +GNaL] *= cvar[1 + (offset*18)];		// GNaL
  CONSTANTS[(offset * num_of_constants) +Gto] *= cvar[2 + (offset*18)];		// Gto
  CONSTANTS[(offset * num_of_constants) +GKr] *= cvar[3 + (offset*18)];		// GKr
  CONSTANTS[(offset * num_of_constants) +GKs] *= cvar[4 + (offset*18)];		// GKs
  CONSTANTS[(offset * num_of_constants) +GK1] *= cvar[5 + (offset*18)];		// GK1
  CONSTANTS[(offset * num_of_constants) +Gncx] *= cvar[6 + (offset*18)];		// GNaCa
  CONSTANTS[(offset * num_of_constants) +GKb] *= cvar[7 + (offset*18)];		// GKb
  CONSTANTS[(offset * num_of_constants) +PCa] *= cvar[8 + (offset*18)];		// PCa
  CONSTANTS[(offset * num_of_constants) +Pnak] *= cvar[9 + (offset*18)];		// INaK
  CONSTANTS[(offset * num_of_constants) +PNab] *= cvar[10 + (offset*18)];		// PNab
  CONSTANTS[(offset * num_of_constants) +PCab] *= cvar[11 + (offset*18)];		// PCab
  CONSTANTS[(offset * num_of_constants) +GpCa] *= cvar[12 + (offset*18)];		// GpCa
  CONSTANTS[(offset * num_of_constants) +KmCaMK] *= cvar[17 + (offset*18)];	// KCaMK

  // Additional constants
  CONSTANTS[(offset * num_of_constants) +Jrel_scale] *= cvar[13 + (offset*18)];	// SERCA_Total (release)
  CONSTANTS[(offset * num_of_constants) +Jup_scale] *= cvar[14 + (offset*18)];	// RyR_Total (uptake)
  CONSTANTS[(offset * num_of_constants) +Jtr_scale] *= cvar[15 + (offset*18)];	// Trans_Total (NSR to JSR translocation)
  CONSTANTS[(offset * num_of_constants) +Jleak_scale] *= cvar[16 + (offset*18)];	// Leak_Total (Ca leak from NSR)
  // CONSTANTS[(offset * num_of_constants) +KCaMK_scale] *= cvar[17 + (offset*18)];	// KCaMK
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset)
{
int num_of_constants = 146;

CONSTANTS[GK1+(offset * num_of_constants)] = CONSTANTS[GK1+(offset * num_of_constants)] * ((ic50[2 + (offset*14)] > epsilon && ic50[3+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[2+ (offset*14)],ic50[3+ (offset*14)])) : 1.);
CONSTANTS[GKr+(offset * num_of_constants)] = CONSTANTS[GKr+(offset * num_of_constants)] * ((ic50[12+ (offset*14)] > epsilon && ic50[13+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[12+ (offset*14)],ic50[13+ (offset*14)])) : 1.);
CONSTANTS[GKs+(offset * num_of_constants)] = CONSTANTS[GKs+(offset * num_of_constants)] * ((ic50[4 + (offset*14)] > epsilon && ic50[5+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[4+ (offset*14)],ic50[5+ (offset*14)])) : 1.);
CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] * ((ic50[8+ (offset*14)] > epsilon && ic50[9+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[8+ (offset*14)],ic50[9+ (offset*14)])) : 1.);
CONSTANTS[GNa+(offset * num_of_constants)] = CONSTANTS[GNa+(offset * num_of_constants)] * ((ic50[6 + (offset*14)] > epsilon && ic50[7+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[6+ (offset*14)],ic50[7+ (offset*14)])) : 1.);
CONSTANTS[Gto+(offset * num_of_constants)] = CONSTANTS[Gto+(offset * num_of_constants)] * ((ic50[10 + (offset*14)] > epsilon && ic50[11+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[10+ (offset*14)],ic50[11+ (offset*14)])) : 1.);
CONSTANTS[PCa+(offset * num_of_constants)] = CONSTANTS[PCa+(offset * num_of_constants)] * ( (ic50[0 + (offset*14)] > epsilon && ic50[1+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[0+ (offset*14)],ic50[1+ (offset*14)])) : 1.);

}

// void initConsts(int offset)
// {
// 	___initConsts(0.,offset);
// }

// void initConsts(double type)
// {
// 	___initConsts(type, offset);
// }

__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, double *cvar, bool is_dutta, bool is_cvar,  int offset)
{
  // int num_of_constants = 146;
  // printf("ic50:%d %lf, %lf, %lf\n",offset,ic50[0 + (offset*14)],ic50[1 + (offset*14)],ic50[2 + (offset*14)]);

	___initConsts(CONSTANTS, STATES, type, offset); // initconst kan minta 
	// // mpi_printf(0,"Celltype: %lf\n", CONSTANTS[celltype]);
	// #ifndef COMPONENT_PATCH // for patch clamp component based research
	// // mpi_printf(0,"Control %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	// #endif
	if(is_dutta == true){
		___applyDutta(CONSTANTS, offset);
	}

  if(is_cvar == true){
		___applyCvar(CONSTANTS, cvar, offset);
	}
	// #ifndef COMPONENT_PATCH
	// // mpi_printf(0,"After Dutta %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	// #endif	
	// ___applyDrugEffect(CONSTANTS, conc, ic50, 10E-14, offset);
	// #ifndef COMPONENT_PATCH
	// // mpi_printf(0,"After drug %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	// #endif


}



__device__ void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int offset )
{
int num_of_constants = 146; //done
int num_of_states = 41; //done
int num_of_algebraic = 199; //done
int num_of_rates = 41; //done

ALGEBRAIC[(offset * num_of_algebraic) +Istim] = (TIME>=CONSTANTS[(offset * num_of_constants) + stim_start] && (TIME - CONSTANTS[(offset * num_of_constants) + stim_start]) - floor((TIME - CONSTANTS[(offset * num_of_constants) + stim_start])/CONSTANTS[(offset * num_of_constants) + BCL])*CONSTANTS[(offset * num_of_constants) + BCL]<=CONSTANTS[(offset * num_of_constants) + duration] ? CONSTANTS[(offset * num_of_constants) + amp] : 0.000000);
ALGEBRAIC[(offset * num_of_algebraic) +hLss] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+87.6100)/7.48800));
ALGEBRAIC[(offset * num_of_algebraic) +hLssp] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+93.8100)/7.48800));
ALGEBRAIC[(offset * num_of_algebraic) +mss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+CONSTANTS[(offset * num_of_constants) + mssV1])/CONSTANTS[(offset * num_of_constants) + mssV2]));
ALGEBRAIC[(offset * num_of_algebraic) +tm] = 1.00000/( CONSTANTS[(offset * num_of_constants) + mtD1]*exp((STATES[(offset * num_of_states) + V]+CONSTANTS[(offset * num_of_constants) + mtV1])/CONSTANTS[(offset * num_of_constants) + mtV2])+ CONSTANTS[(offset * num_of_constants) + mtD2]*exp(- (STATES[(offset * num_of_states) + V]+CONSTANTS[(offset * num_of_constants) + mtV3])/CONSTANTS[(offset * num_of_constants) + mtV4]));
ALGEBRAIC[(offset * num_of_algebraic) +hss] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+CONSTANTS[(offset * num_of_constants) + hssV1])/CONSTANTS[(offset * num_of_constants) + hssV2]));
ALGEBRAIC[(offset * num_of_algebraic) +thf] = 1.00000/( 1.43200e-05*exp(- (STATES[(offset * num_of_states) + V]+1.19600)/6.28500)+ 6.14900*exp((STATES[(offset * num_of_states) + V]+0.509600)/20.2700));
ALGEBRAIC[(offset * num_of_algebraic) +ths] = 1.00000/( 0.00979400*exp(- (STATES[(offset * num_of_states) + V]+17.9500)/28.0500)+ 0.334300*exp((STATES[(offset * num_of_states) + V]+5.73000)/56.6600));
ALGEBRAIC[(offset * num_of_algebraic) +ass] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 14.3400)/14.8200));
ALGEBRAIC[(offset * num_of_algebraic) +ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(offset * num_of_states) + V]+100.000)/29.3814)));
ALGEBRAIC[(offset * num_of_algebraic) +dss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+3.94000)/4.23000));
ALGEBRAIC[(offset * num_of_algebraic) +td] = 0.600000+1.00000/(exp( - 0.0500000*(STATES[(offset * num_of_states) + V]+6.00000))+exp( 0.0900000*(STATES[(offset * num_of_states) + V]+14.0000)));
ALGEBRAIC[(offset * num_of_algebraic) + fss] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+19.5800)/3.69600));
ALGEBRAIC[(offset * num_of_algebraic) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(offset * num_of_states) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(offset * num_of_states) + V]+20.0000)/10.0000));
ALGEBRAIC[(offset * num_of_algebraic) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(offset * num_of_states) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(offset * num_of_states) + V]+5.00000)/6.00000));
ALGEBRAIC[(offset * num_of_algebraic) + fcass] = ALGEBRAIC[(offset * num_of_algebraic) + fss];
ALGEBRAIC[(offset * num_of_algebraic) + km2n] =  STATES[(offset * num_of_states) + jca]*1.00000;
ALGEBRAIC[(offset * num_of_algebraic) + anca] = 1.00000/(CONSTANTS[(offset * num_of_constants) + k2n]/ALGEBRAIC[(offset * num_of_algebraic) + km2n]+pow(1.00000+CONSTANTS[(offset * num_of_constants) + Kmn]/STATES[(offset * num_of_states) + cass], 4.00000));
ALGEBRAIC[(offset * num_of_algebraic) + xrss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+8.33700)/6.78900));
ALGEBRAIC[(offset * num_of_algebraic) + txrf] = 12.9800+1.00000/( 0.365200*exp((STATES[(offset * num_of_states) + V] - 31.6600)/3.86900)+ 4.12300e-05*exp(- (STATES[(offset * num_of_states) + V] - 47.7800)/20.3800));
ALGEBRAIC[(offset * num_of_algebraic) + txrs] = 1.86500+1.00000/( 0.0662900*exp((STATES[(offset * num_of_states) + V] - 34.7000)/7.35500)+ 1.12800e-05*exp(- (STATES[(offset * num_of_states) + V] - 29.7400)/25.9400));
ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+11.6000)/8.93200));
ALGEBRAIC[(offset * num_of_algebraic) + txs1] = 817.300+1.00000/( 0.000232600*exp((STATES[(offset * num_of_states) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(offset * num_of_states) + V]+210.000)/230.000));
ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+ 2.55380*CONSTANTS[(offset * num_of_constants) + ko]+144.590)/( 1.56920*CONSTANTS[(offset * num_of_constants) + ko]+3.81150)));
ALGEBRAIC[(offset * num_of_algebraic) + txk1] = 122.200/(exp(- (STATES[(offset * num_of_states) + V]+127.200)/20.3600)+exp((STATES[(offset * num_of_states) + V]+236.800)/69.3300));
ALGEBRAIC[(offset * num_of_algebraic) + jss] = ALGEBRAIC[(offset * num_of_algebraic) + hss];
ALGEBRAIC[(offset * num_of_algebraic) + tj] = 2.03800+1.00000/( 0.0213600*exp(- (STATES[(offset * num_of_states) + V]+100.600)/8.28100)+ 0.305200*exp((STATES[(offset * num_of_states) + V]+0.994100)/38.4500));
ALGEBRAIC[(offset * num_of_algebraic) + assp] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 24.3400)/14.8200));
ALGEBRAIC[(offset * num_of_algebraic) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(offset * num_of_states) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(offset * num_of_states) + V] - 4.00000)/7.00000));
ALGEBRAIC[(offset * num_of_algebraic) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(offset * num_of_states) + V]/3.00000)+ 0.000120000*exp(STATES[(offset * num_of_states) + V]/7.00000));
ALGEBRAIC[(offset * num_of_algebraic) + tffp] =  2.50000*ALGEBRAIC[(offset * num_of_algebraic) + tff];
ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] = ALGEBRAIC[(offset * num_of_algebraic) + xs1ss];
ALGEBRAIC[(offset * num_of_algebraic) + txs2] = 1.00000/( 0.0100000*exp((STATES[(offset * num_of_states) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(offset * num_of_states) + V]+66.5400)/31.0000));
ALGEBRAIC[(offset * num_of_algebraic) + CaMKb] = ( CONSTANTS[(offset * num_of_constants) + CaMKo]*(1.00000 - STATES[(offset * num_of_states) + CaMKt]))/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaM]/STATES[(offset * num_of_states) + cass]);
ALGEBRAIC[(offset * num_of_algebraic) + hssp] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+89.1000)/6.08600));
ALGEBRAIC[(offset * num_of_algebraic) + thsp] =  3.00000*ALGEBRAIC[(offset * num_of_algebraic) + ths];
ALGEBRAIC[(offset * num_of_algebraic) + tjp] =  1.46000*ALGEBRAIC[(offset * num_of_algebraic) + tj];
ALGEBRAIC[(offset * num_of_algebraic) + mLss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+42.8500)/5.26400));
ALGEBRAIC[(offset * num_of_algebraic) + tmL] = ALGEBRAIC[(offset * num_of_algebraic) + tm];
ALGEBRAIC[(offset * num_of_algebraic) + tfcafp] =  2.50000*ALGEBRAIC[(offset * num_of_algebraic) + tfcaf];
ALGEBRAIC[(offset * num_of_algebraic) + iss] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+43.9400)/5.71100));
ALGEBRAIC[(offset * num_of_algebraic) + delta_epi] = (CONSTANTS[(offset * num_of_constants) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(offset * num_of_states) + V]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(offset * num_of_states) + V]+100.000)/100.000)+ 0.0800400*exp((STATES[(offset * num_of_states) + V]+50.0000)/16.5900));
ALGEBRAIC[(offset * num_of_algebraic) + tiF] =  ALGEBRAIC[(offset * num_of_algebraic) + tiF_b]*ALGEBRAIC[(offset * num_of_algebraic) + delta_epi];
ALGEBRAIC[(offset * num_of_algebraic) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(offset * num_of_states) + V]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(offset * num_of_states) + V]+114.100)/8.07900));
ALGEBRAIC[(offset * num_of_algebraic) + tiS] =  ALGEBRAIC[(offset * num_of_algebraic) + tiS_b]*ALGEBRAIC[(offset * num_of_algebraic) + delta_epi];
ALGEBRAIC[(offset * num_of_algebraic) + dti_develop] = 1.35400+0.000100000/(exp((STATES[(offset * num_of_states) + V] - 167.400)/15.8900)+exp(- (STATES[(offset * num_of_states) + V] - 12.2300)/0.215400));
ALGEBRAIC[(offset * num_of_algebraic) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(offset * num_of_states) + V]+70.0000)/20.0000));
ALGEBRAIC[(offset * num_of_algebraic) + tiFp] =  ALGEBRAIC[(offset * num_of_algebraic) + dti_develop]*ALGEBRAIC[(offset * num_of_algebraic) + dti_recover]*ALGEBRAIC[(offset * num_of_algebraic) + tiF];
ALGEBRAIC[(offset * num_of_algebraic) + tiSp] =  ALGEBRAIC[(offset * num_of_algebraic) + dti_develop]*ALGEBRAIC[(offset * num_of_algebraic) + dti_recover]*ALGEBRAIC[(offset * num_of_algebraic) + tiS];
ALGEBRAIC[(offset * num_of_algebraic) + f] =  CONSTANTS[(offset * num_of_constants) + Aff]*STATES[(offset * num_of_states) + ff]+ CONSTANTS[(offset * num_of_constants) + Afs]*STATES[(offset * num_of_states) + fs];
ALGEBRAIC[(offset * num_of_algebraic) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(offset * num_of_states) + V] - 10.0000)/10.0000));
ALGEBRAIC[(offset * num_of_algebraic) + Afcas] = 1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + Afcaf];
ALGEBRAIC[(offset * num_of_algebraic) + fca] =  ALGEBRAIC[(offset * num_of_algebraic) + Afcaf]*STATES[(offset * num_of_states) + fcaf]+ ALGEBRAIC[(offset * num_of_algebraic) + Afcas]*STATES[(offset * num_of_states) + fcas];
ALGEBRAIC[(offset * num_of_algebraic) + fp] =  CONSTANTS[(offset * num_of_constants) + Aff]*STATES[(offset * num_of_states) + ffp]+ CONSTANTS[(offset * num_of_constants) + Afs]*STATES[(offset * num_of_states) + fs];
ALGEBRAIC[(offset * num_of_algebraic) + fcap] =  ALGEBRAIC[(offset * num_of_algebraic) + Afcaf]*STATES[(offset * num_of_states) + fcafp]+ ALGEBRAIC[(offset * num_of_algebraic) + Afcas]*STATES[(offset * num_of_states) + fcas];
ALGEBRAIC[(offset * num_of_algebraic) + vffrt] = ( STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]);
ALGEBRAIC[(offset * num_of_algebraic) + vfrt] = ( STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]);
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL] = ( 4.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( STATES[(offset * num_of_states) + cass]*exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.341000*CONSTANTS[(offset * num_of_constants) + cao]))/(exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + CaMKa] = ALGEBRAIC[(offset * num_of_algebraic) + CaMKb]+STATES[(offset * num_of_states) + CaMKt];
ALGEBRAIC[(offset * num_of_algebraic) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + ICaL] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCa]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCap]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp] = ( CONSTANTS[(offset * num_of_constants) + a_rel]*- ALGEBRAIC[(offset * num_of_algebraic) + ICaL])/(1.00000+ 1.00000*pow(1.50000/STATES[(offset * num_of_states) + cajsr], 8.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] = (CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp]*1.70000 : ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp] = CONSTANTS[(offset * num_of_constants) + bt]/(1.00000+0.0123000/STATES[(offset * num_of_states) + cajsr]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_rel] = (ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp] = ( CONSTANTS[(offset * num_of_constants) + a_relp]*- ALGEBRAIC[(offset * num_of_algebraic) + ICaL])/(1.00000+pow(1.50000/STATES[(offset * num_of_states) + cajsr], 8.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] = (CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp]*1.70000 : ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp] = CONSTANTS[(offset * num_of_constants) + btp]/(1.00000+0.0123000/STATES[(offset * num_of_states) + cajsr]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_relp] = (ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + EK] =  (( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T])/CONSTANTS[(offset * num_of_constants) + F])*log(CONSTANTS[(offset * num_of_constants) + ko]/STATES[(offset * num_of_states) + ki]);
ALGEBRAIC[(offset * num_of_algebraic) + AiF] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V] - 213.600)/151.200));
ALGEBRAIC[(offset * num_of_algebraic) + AiS] = 1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + AiF];
ALGEBRAIC[(offset * num_of_algebraic) + i] =  ALGEBRAIC[(offset * num_of_algebraic) + AiF]*STATES[(offset * num_of_states) + iF]+ ALGEBRAIC[(offset * num_of_algebraic) + AiS]*STATES[(offset * num_of_states) + iS];
ALGEBRAIC[(offset * num_of_algebraic) + ip] =  ALGEBRAIC[(offset * num_of_algebraic) + AiF]*STATES[(offset * num_of_states) + iFp]+ ALGEBRAIC[(offset * num_of_algebraic) + AiS]*STATES[(offset * num_of_states) + iSp];
ALGEBRAIC[(offset * num_of_algebraic) + fItop] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Ito] =  CONSTANTS[(offset * num_of_constants) + Gto]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK])*( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fItop])*STATES[(offset * num_of_states) + a]*ALGEBRAIC[(offset * num_of_algebraic) + i]+ ALGEBRAIC[(offset * num_of_algebraic) + fItop]*STATES[(offset * num_of_states) + ap]*ALGEBRAIC[(offset * num_of_algebraic) + ip]);
ALGEBRAIC[(offset * num_of_algebraic) + Axrf] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+54.8100)/38.2100));
ALGEBRAIC[(offset * num_of_algebraic) + Axrs] = 1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + Axrf];
ALGEBRAIC[(offset * num_of_algebraic) + xr] =  ALGEBRAIC[(offset * num_of_algebraic) + Axrf]*STATES[(offset * num_of_states) + xrf]+ ALGEBRAIC[(offset * num_of_algebraic) + Axrs]*STATES[(offset * num_of_states) + xrs];
ALGEBRAIC[(offset * num_of_algebraic) + rkr] = ( (1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+55.0000)/75.0000)))*1.00000)/(1.00000+exp((STATES[(offset * num_of_states) + V] - 10.0000)/30.0000));
ALGEBRAIC[(offset * num_of_algebraic) + IKr] =  CONSTANTS[(offset * num_of_constants) + GKr]* pow((CONSTANTS[(offset * num_of_constants) + ko]/5.40000), 1.0 / 2)*ALGEBRAIC[(offset * num_of_algebraic) + xr]*ALGEBRAIC[(offset * num_of_algebraic) + rkr]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + EKs] =  (( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T])/CONSTANTS[(offset * num_of_constants) + F])*log((CONSTANTS[(offset * num_of_constants) + ko]+ CONSTANTS[(offset * num_of_constants) + PKNa]*CONSTANTS[(offset * num_of_constants) + nao])/(STATES[(offset * num_of_states) + ki]+ CONSTANTS[(offset * num_of_constants) + PKNa]*STATES[(offset * num_of_states) + nai]));
ALGEBRAIC[(offset * num_of_algebraic) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(offset * num_of_states) + cai], 1.40000));
ALGEBRAIC[(offset * num_of_algebraic) + IKs] =  CONSTANTS[(offset * num_of_constants) + GKs]*ALGEBRAIC[(offset * num_of_algebraic) + KsCa]*STATES[(offset * num_of_states) + xs1]*STATES[(offset * num_of_states) + xs2]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EKs]);
ALGEBRAIC[(offset * num_of_algebraic) + rk1] = 1.00000/(1.00000+exp(((STATES[(offset * num_of_states) + V]+105.800) -  2.60000*CONSTANTS[(offset * num_of_constants) + ko])/9.49300));
ALGEBRAIC[(offset * num_of_algebraic) + IK1] =  CONSTANTS[(offset * num_of_constants) + GK1]* pow(CONSTANTS[(offset * num_of_constants) + ko], 1.0 / 2)*ALGEBRAIC[(offset * num_of_algebraic) + rk1]*STATES[(offset * num_of_states) + xk1]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + Knao] =  CONSTANTS[(offset * num_of_constants) + Knao0]*exp(( (1.00000 - CONSTANTS[(offset * num_of_constants) + delta])*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( 3.00000*CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + a3] = ( CONSTANTS[(offset * num_of_constants) + k3p]*pow(CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + P] = CONSTANTS[(offset * num_of_constants) + eP]/(1.00000+CONSTANTS[(offset * num_of_constants) + H]/CONSTANTS[(offset * num_of_constants) + Khp]+STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + Knap]+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kxkur]);
ALGEBRAIC[(offset * num_of_algebraic) + b3] = ( CONSTANTS[(offset * num_of_constants) + k3m]*ALGEBRAIC[(offset * num_of_algebraic) + P]*CONSTANTS[(offset * num_of_constants) + H])/(1.00000+CONSTANTS[(offset * num_of_constants) + MgATP]/CONSTANTS[(offset * num_of_constants) + Kmgatp]);
ALGEBRAIC[(offset * num_of_algebraic) + Knai] =  CONSTANTS[(offset * num_of_constants) + Knai0]*exp(( CONSTANTS[(offset * num_of_constants) + delta]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( 3.00000*CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + a1] = ( CONSTANTS[(offset * num_of_constants) + k1p]*pow(STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000))/((pow(1.00000+STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000)+pow(1.00000+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + b2] = ( CONSTANTS[(offset * num_of_constants) + k2m]*pow(CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + b4] = ( CONSTANTS[(offset * num_of_constants) + k4m]*pow(STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000))/((pow(1.00000+STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000)+pow(1.00000+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + x1] =  CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]+ CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2];
ALGEBRAIC[(offset * num_of_algebraic) + x2] =  ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]*ALGEBRAIC[(offset * num_of_algebraic) + b4]+ ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + b1]*ALGEBRAIC[(offset * num_of_algebraic) + b4]+ CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]*ALGEBRAIC[(offset * num_of_algebraic) + b4];
ALGEBRAIC[(offset * num_of_algebraic) + x3] =  CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]*CONSTANTS[(offset * num_of_constants) + a4]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]*CONSTANTS[(offset * num_of_constants) + b1];
ALGEBRAIC[(offset * num_of_algebraic) + x4] =  ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]*ALGEBRAIC[(offset * num_of_algebraic) + a1];
ALGEBRAIC[(offset * num_of_algebraic) + E1] = ALGEBRAIC[(offset * num_of_algebraic) + x1]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + E2] = ALGEBRAIC[(offset * num_of_algebraic) + x2]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + JnakNa] =  3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E1]*ALGEBRAIC[(offset * num_of_algebraic) + a3] -  ALGEBRAIC[(offset * num_of_algebraic) + E2]*ALGEBRAIC[(offset * num_of_algebraic) + b3]);
ALGEBRAIC[(offset * num_of_algebraic) + E3] = ALGEBRAIC[(offset * num_of_algebraic) + x3]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + E4] = ALGEBRAIC[(offset * num_of_algebraic) + x4]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + JnakK] =  2.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4]*CONSTANTS[(offset * num_of_constants) + b1] -  ALGEBRAIC[(offset * num_of_algebraic) + E3]*ALGEBRAIC[(offset * num_of_algebraic) + a1]);
ALGEBRAIC[(offset * num_of_algebraic) + INaK] =  CONSTANTS[(offset * num_of_constants) + Pnak]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JnakNa]+ CONSTANTS[(offset * num_of_constants) + zk]*ALGEBRAIC[(offset * num_of_algebraic) + JnakK]);
ALGEBRAIC[(offset * num_of_algebraic) + xkb] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 14.4800)/18.3400));
ALGEBRAIC[(offset * num_of_algebraic) + IKb] =  CONSTANTS[(offset * num_of_constants) + GKb]*ALGEBRAIC[(offset * num_of_algebraic) + xkb]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + JdiffK] = (STATES[(offset * num_of_states) + kss] - STATES[(offset * num_of_states) + ki])/2.00000;
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK] = ( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( 0.750000*STATES[(offset * num_of_states) + kss]*exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.750000*CONSTANTS[(offset * num_of_constants) + ko]))/(exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + ICaK] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCaK]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCaKp]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + ENa] =  (( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T])/CONSTANTS[(offset * num_of_constants) + F])*log(CONSTANTS[(offset * num_of_constants) + nao]/STATES[(offset * num_of_states) + nai]);
ALGEBRAIC[(offset * num_of_algebraic) + h] =  CONSTANTS[(offset * num_of_constants) + Ahf]*STATES[(offset * num_of_states) + hf]+ CONSTANTS[(offset * num_of_constants) + Ahs]*STATES[(offset * num_of_states) + hs];
ALGEBRAIC[(offset * num_of_algebraic) + hp] =  CONSTANTS[(offset * num_of_constants) + Ahf]*STATES[(offset * num_of_states) + hf]+ CONSTANTS[(offset * num_of_constants) + Ahs]*STATES[(offset * num_of_states) + hsp];
ALGEBRAIC[(offset * num_of_algebraic) + fINap] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + INa] =  CONSTANTS[(offset * num_of_constants) + GNa]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + ENa])*pow(STATES[(offset * num_of_states) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fINap])*ALGEBRAIC[(offset * num_of_algebraic) + h]*STATES[(offset * num_of_states) + j]+ ALGEBRAIC[(offset * num_of_algebraic) + fINap]*ALGEBRAIC[(offset * num_of_algebraic) + hp]*STATES[(offset * num_of_states) + jp]);
ALGEBRAIC[(offset * num_of_algebraic) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + INaL] =  CONSTANTS[(offset * num_of_constants) + GNaL]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + ENa])*STATES[(offset * num_of_states) + mL]*( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fINaLp])*STATES[(offset * num_of_states) + hL]+ ALGEBRAIC[(offset * num_of_algebraic) + fINaLp]*STATES[(offset * num_of_states) + hLp]);
ALGEBRAIC[(offset * num_of_algebraic) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(offset * num_of_constants) + KmCaAct]/STATES[(offset * num_of_states) + cai], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + hna] = exp(( CONSTANTS[(offset * num_of_constants) + qna]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + h7_i] = 1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+1.00000/ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h8_i] = CONSTANTS[(offset * num_of_constants) + nao]/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + hna]*ALGEBRAIC[(offset * num_of_algebraic) + h7_i]);
ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_i]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + h1_i] = 1.00000+ (STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h2_i] = ( STATES[(offset * num_of_states) + nai]*ALGEBRAIC[(offset * num_of_algebraic) + hna])/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + h1_i]);
ALGEBRAIC[(offset * num_of_algebraic) + k4pp_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h2_i]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + h4_i] = 1.00000+ (STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h5_i] = ( STATES[(offset * num_of_states) + nai]*STATES[(offset * num_of_states) + nai])/( ALGEBRAIC[(offset * num_of_algebraic) + h4_i]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + k7_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h5_i]*ALGEBRAIC[(offset * num_of_algebraic) + h2_i]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + k8_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_i]*CONSTANTS[(offset * num_of_constants) + h11_i]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + h9_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h7_i];
ALGEBRAIC[(offset * num_of_algebraic) + k3p_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h9_i]*CONSTANTS[(offset * num_of_constants) + wca];
ALGEBRAIC[(offset * num_of_algebraic) + k3_i] = ALGEBRAIC[(offset * num_of_algebraic) + k3p_i]+ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + hca] = exp(( CONSTANTS[(offset * num_of_constants) + qca]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + h3_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) +  h1_i];
ALGEBRAIC[(offset * num_of_algebraic) + k4p_i] = ( ALGEBRAIC[(offset * num_of_algebraic) +  h3_i]*CONSTANTS[(offset * num_of_constants) + wca])/ALGEBRAIC[(offset * num_of_algebraic) +  hca];
ALGEBRAIC[(offset * num_of_algebraic) + k4_i] = ALGEBRAIC[(offset * num_of_algebraic) +  k4p_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k4pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + h6_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) +  h4_i];
ALGEBRAIC[(offset * num_of_algebraic) + k6_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h6_i]*STATES[(offset * num_of_states) + cai]*CONSTANTS[(offset * num_of_constants) + kcaon];
ALGEBRAIC[(offset * num_of_algebraic) + x1_i] =  CONSTANTS[(offset * num_of_constants) + k2_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k4_i]*(ALGEBRAIC[(offset * num_of_algebraic) +  k7_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k6_i])+ CONSTANTS[(offset * num_of_constants) + k5_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k7_i]*(CONSTANTS[(offset * num_of_constants) + k2_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k3_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x2_i] =  CONSTANTS[(offset * num_of_constants) + k1_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k7_i]*(ALGEBRAIC[(offset * num_of_algebraic) +  k4_i]+CONSTANTS[(offset * num_of_constants) + k5_i])+ ALGEBRAIC[(offset * num_of_algebraic) +  k4_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k6_i]*(CONSTANTS[(offset * num_of_constants) + k1_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k8_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x3_i] =  CONSTANTS[(offset * num_of_constants) + k1_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k3_i]*(ALGEBRAIC[(offset * num_of_algebraic) +  k7_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k6_i])+ ALGEBRAIC[(offset * num_of_algebraic) +  k8_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k6_i]*(CONSTANTS[(offset * num_of_constants) + k2_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k3_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x4_i] =  CONSTANTS[(offset * num_of_constants) + k2_i]*ALGEBRAIC[(offset * num_of_algebraic) +  k8_i]*(ALGEBRAIC[(offset * num_of_algebraic) +  k4_i]+CONSTANTS[(offset * num_of_constants) + k5_i])+ ALGEBRAIC[(offset * num_of_algebraic) +  k3_i]*CONSTANTS[(offset * num_of_constants) + k5_i]*(CONSTANTS[(offset * num_of_constants) + k1_i]+ALGEBRAIC[(offset * num_of_algebraic) +  k8_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E1_i] = ALGEBRAIC[(offset * num_of_algebraic) +  x1_i]/(ALGEBRAIC[(offset * num_of_algebraic) +  x1_i]+ALGEBRAIC[(offset * num_of_algebraic) +  x2_i]+ALGEBRAIC[(offset * num_of_algebraic) +  x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E2_i] = ALGEBRAIC[(offset * num_of_algebraic) + x2_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E3_i] = ALGEBRAIC[(offset * num_of_algebraic) + x3_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E4_i] = ALGEBRAIC[(offset * num_of_algebraic) + x4_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4_i]*ALGEBRAIC[(offset * num_of_algebraic) + k7_i] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_i]*ALGEBRAIC[(offset * num_of_algebraic) + k8_i])+ ALGEBRAIC[(offset * num_of_algebraic) + E3_i]*ALGEBRAIC[(offset * num_of_algebraic) + k4pp_i]) -  ALGEBRAIC[(offset * num_of_algebraic) + E2_i]*ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_i] =  ALGEBRAIC[(offset * num_of_algebraic) + E2_i]*CONSTANTS[(offset * num_of_constants) + k2_i] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_i]*CONSTANTS[(offset * num_of_constants) + k1_i];
ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i] =  0.800000*CONSTANTS[(offset * num_of_constants) + Gncx]*ALGEBRAIC[(offset * num_of_algebraic) + allo_i]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_i]+ CONSTANTS[(offset * num_of_constants) + zca]*ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_i]);
ALGEBRAIC[(offset * num_of_algebraic) + INab] = ( CONSTANTS[(offset * num_of_constants) + PNab]*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( STATES[(offset * num_of_states) + nai]*exp(ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - CONSTANTS[(offset * num_of_constants) + nao]))/(exp(ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa] = (STATES[(offset * num_of_states) + nass] - STATES[(offset * num_of_states) + nai])/2.00000;
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa] = ( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( 0.750000*STATES[(offset * num_of_states) + nass]*exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.750000*CONSTANTS[(offset * num_of_constants) + nao]))/(exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + ICaNa] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCaNa]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCaNap]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(offset * num_of_constants) + KmCaAct]/STATES[(offset * num_of_states) + cass], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + h7_ss] = 1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+1.00000/ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h8_ss] = CONSTANTS[(offset * num_of_constants) + nao]/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + hna]*ALGEBRAIC[(offset * num_of_algebraic) + h7_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_ss]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + h1_ss] = 1.00000+ (STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h2_ss] = ( STATES[(offset * num_of_states) + nass]*ALGEBRAIC[(offset * num_of_algebraic) + hna])/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + h1_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h2_ss]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + h4_ss] = 1.00000+ (STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h5_ss] = ( STATES[(offset * num_of_states) + nass]*STATES[(offset * num_of_states) + nass])/( ALGEBRAIC[(offset * num_of_algebraic) + h4_ss]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + k7_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h5_ss]*ALGEBRAIC[(offset * num_of_algebraic) + h2_ss]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + k8_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_ss]*CONSTANTS[(offset * num_of_constants) + h11_ss]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + h9_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h7_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k3p_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h9_ss]*CONSTANTS[(offset * num_of_constants) + wca];
ALGEBRAIC[(offset * num_of_algebraic) + k3_ss] = ALGEBRAIC[(offset * num_of_algebraic) + k3p_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + h3_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h1_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k4p_ss] = ( ALGEBRAIC[(offset * num_of_algebraic) + h3_ss]*CONSTANTS[(offset * num_of_constants) + wca])/ALGEBRAIC[(offset * num_of_algebraic) + hca];
ALGEBRAIC[(offset * num_of_algebraic) + k4_ss] = ALGEBRAIC[(offset * num_of_algebraic) + k4p_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + h6_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h4_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k6_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h6_ss]*STATES[(offset * num_of_states) + cass]*CONSTANTS[(offset * num_of_constants) + kcaon];
ALGEBRAIC[(offset * num_of_algebraic) + x1_ss] =  CONSTANTS[(offset * num_of_constants) + k2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k6_ss])+ CONSTANTS[(offset * num_of_constants) + k5_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]*(CONSTANTS[(offset * num_of_constants) + k2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x2_ss] =  CONSTANTS[(offset * num_of_constants) + k1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]+CONSTANTS[(offset * num_of_constants) + k5_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k6_ss]*(CONSTANTS[(offset * num_of_constants) + k1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x3_ss] =  CONSTANTS[(offset * num_of_constants) + k1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k6_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k6_ss]*(CONSTANTS[(offset * num_of_constants) + k2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x4_ss] =  CONSTANTS[(offset * num_of_constants) + k2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]+CONSTANTS[(offset * num_of_constants) + k5_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]*CONSTANTS[(offset * num_of_constants) + k5_ss]*(CONSTANTS[(offset * num_of_constants) + k1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E1_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E2_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E3_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E4_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k8_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + E3_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss]) -  ALGEBRAIC[(offset * num_of_algebraic) + E2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + E2_ss]*CONSTANTS[(offset * num_of_constants) + k2_ss] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_ss]*CONSTANTS[(offset * num_of_constants) + k1_ss];
ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss] =  0.200000*CONSTANTS[(offset * num_of_constants) + Gncx]*ALGEBRAIC[(offset * num_of_algebraic) + allo_ss]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_ss]+ CONSTANTS[(offset * num_of_constants) + zca]*ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + IpCa] = ( CONSTANTS[(offset * num_of_constants) + GpCa]*STATES[(offset * num_of_states) + cai])/(CONSTANTS[(offset * num_of_constants) + KmCap]+STATES[(offset * num_of_states) + cai]);
ALGEBRAIC[(offset * num_of_algebraic) + ICab] = ( CONSTANTS[(offset * num_of_constants) + PCab]*4.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( STATES[(offset * num_of_states) + cai]*exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.341000*CONSTANTS[(offset * num_of_constants) + cao]))/(exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + Jdiff] = (STATES[(offset * num_of_states) + cass] - STATES[(offset * num_of_states) + cai])/0.200000;
ALGEBRAIC[(offset * num_of_algebraic) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fJrelp])*STATES[(offset * num_of_states) + Jrelnp]+ ALGEBRAIC[(offset * num_of_algebraic) + fJrelp]*STATES[(offset * num_of_states) + Jrelp];
ALGEBRAIC[(offset * num_of_algebraic) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + BSRmax]*CONSTANTS[(offset * num_of_constants) + KmBSR])/pow(CONSTANTS[(offset * num_of_constants) + KmBSR]+STATES[(offset * num_of_states) + cass], 2.00000)+( CONSTANTS[(offset * num_of_constants) + BSLmax]*CONSTANTS[(offset * num_of_constants) + KmBSL])/pow(CONSTANTS[(offset * num_of_constants) + KmBSL]+STATES[(offset * num_of_states) + cass], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jupnp] = ( CONSTANTS[(offset * num_of_constants) + upScale]*0.00437500*STATES[(offset * num_of_states) + cai])/(STATES[(offset * num_of_states) + cai]+0.000920000);
ALGEBRAIC[(offset * num_of_algebraic) + Jupp] = ( CONSTANTS[(offset * num_of_constants) + upScale]*2.75000*0.00437500*STATES[(offset * num_of_states) + cai])/((STATES[(offset * num_of_states) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(offset * num_of_algebraic) + fJupp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Jleak] = ( 0.00393750*STATES[(offset * num_of_states) + cansr])/15.0000;
ALGEBRAIC[(offset * num_of_algebraic) + Jup] = ( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fJupp])*ALGEBRAIC[(offset * num_of_algebraic) + Jupnp]+ ALGEBRAIC[(offset * num_of_algebraic) + fJupp]*ALGEBRAIC[(offset * num_of_algebraic) + Jupp]) - ALGEBRAIC[(offset * num_of_algebraic) + Jleak];
ALGEBRAIC[(offset * num_of_algebraic) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + cmdnmax]*CONSTANTS[(offset * num_of_constants) + kmcmdn])/pow(CONSTANTS[(offset * num_of_constants) + kmcmdn]+STATES[(offset * num_of_states) + cai], 2.00000)+( CONSTANTS[(offset * num_of_constants) + trpnmax]*CONSTANTS[(offset * num_of_constants) + kmtrpn])/pow(CONSTANTS[(offset * num_of_constants) + kmtrpn]+STATES[(offset * num_of_states) + cai], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jtr] = (STATES[(offset * num_of_states) + cansr] - STATES[(offset * num_of_states) + cajsr])/100.000;
ALGEBRAIC[(offset * num_of_algebraic) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + csqnmax]*CONSTANTS[(offset * num_of_constants) + kmcsqn])/pow(CONSTANTS[(offset * num_of_constants) + kmcsqn]+STATES[(offset * num_of_states) + cajsr], 2.00000));

RATES[(offset * num_of_rates) + hL] = (ALGEBRAIC[(offset * num_of_algebraic) + hLss] - STATES[(offset * num_of_states) + hL])/CONSTANTS[(offset * num_of_constants) + thL];
RATES[(offset * num_of_rates) + hLp] = (ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - STATES[(offset * num_of_states) + hLp])/CONSTANTS[(offset * num_of_constants) + thLp];
RATES[(offset * num_of_rates) + m] = (ALGEBRAIC[(offset * num_of_algebraic) + mss] - STATES[(offset * num_of_states) + m])/ALGEBRAIC[(offset * num_of_algebraic) + tm];
RATES[(offset * num_of_rates) + hf] = (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hf])/ALGEBRAIC[(offset * num_of_algebraic) + thf];
RATES[(offset * num_of_rates) + hs] = (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hs])/ALGEBRAIC[(offset * num_of_algebraic) + ths];
RATES[(offset * num_of_rates) + a] = (ALGEBRAIC[(offset * num_of_algebraic) + ass] - STATES[(offset * num_of_states) + a])/ALGEBRAIC[(offset * num_of_algebraic) + ta];
RATES[(offset * num_of_rates) + d] = (ALGEBRAIC[(offset * num_of_algebraic) + dss] - STATES[(offset * num_of_states) + d])/ALGEBRAIC[(offset * num_of_algebraic) + td];
RATES[(offset * num_of_rates) + ff] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ff])/ALGEBRAIC[(offset * num_of_algebraic) + tff];
RATES[(offset * num_of_rates) + fs] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + fs])/ALGEBRAIC[(offset * num_of_algebraic) + tfs];
RATES[(offset * num_of_rates) + jca] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + jca])/CONSTANTS[(offset * num_of_constants) + tjca];
RATES[(offset * num_of_rates) + nca] =  ALGEBRAIC[(offset * num_of_algebraic) + anca]*CONSTANTS[(offset * num_of_constants) + k2n] - STATES[(offset * num_of_states) + nca]*ALGEBRAIC[(offset * num_of_algebraic) + km2n];
RATES[(offset * num_of_rates) + xrf] = (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrf])/ALGEBRAIC[(offset * num_of_algebraic) + txrf];
RATES[(offset * num_of_rates) + xrs] = (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrs])/ALGEBRAIC[(offset * num_of_algebraic) + txrs];
RATES[(offset * num_of_rates) + xs1] = (ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - STATES[(offset * num_of_states) + xs1])/ALGEBRAIC[(offset * num_of_algebraic) + txs1];
RATES[(offset * num_of_rates) + xk1] = (ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - STATES[(offset * num_of_states) + xk1])/ALGEBRAIC[(offset * num_of_algebraic) + txk1];
RATES[(offset * num_of_rates) + j] = (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + j])/ALGEBRAIC[(offset * num_of_algebraic) + tj];
RATES[(offset * num_of_rates) + ap] = (ALGEBRAIC[(offset * num_of_algebraic) + assp] - STATES[(offset * num_of_states) + ap])/ALGEBRAIC[(offset * num_of_algebraic) + ta];
RATES[(offset * num_of_rates) + fcaf] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcaf])/ALGEBRAIC[(offset * num_of_algebraic) + tfcaf];
RATES[(offset * num_of_rates) + fcas] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcas])/ALGEBRAIC[(offset * num_of_algebraic) + tfcas];
RATES[(offset * num_of_rates) + ffp] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ffp])/ALGEBRAIC[(offset * num_of_algebraic) + tffp];
RATES[(offset * num_of_rates) + xs2] = (ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - STATES[(offset * num_of_states) + xs2])/ALGEBRAIC[(offset * num_of_algebraic) + txs2];
RATES[(offset * num_of_rates) + CaMKt] =  CONSTANTS[(offset * num_of_constants) + aCaMK]*ALGEBRAIC[(offset * num_of_algebraic) + CaMKb]*(ALGEBRAIC[(offset * num_of_algebraic) + CaMKb]+STATES[(offset * num_of_states) + CaMKt]) -  CONSTANTS[(offset * num_of_constants) + bCaMK]*STATES[(offset * num_of_states) + CaMKt];
RATES[(offset * num_of_rates) + hsp] = (ALGEBRAIC[(offset * num_of_algebraic) + hssp] - STATES[(offset * num_of_states) + hsp])/ALGEBRAIC[(offset * num_of_algebraic) + thsp];
RATES[(offset * num_of_rates) + jp] = (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + jp])/ALGEBRAIC[(offset * num_of_algebraic) + tjp];
RATES[(offset * num_of_rates) + mL] = (ALGEBRAIC[(offset * num_of_algebraic) + mLss] - STATES[(offset * num_of_states) + mL])/ALGEBRAIC[(offset * num_of_algebraic) + tmL];
RATES[(offset * num_of_rates) + fcafp] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcafp])/ALGEBRAIC[(offset * num_of_algebraic) + tfcafp];
RATES[(offset * num_of_rates) + iF] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iF])/ALGEBRAIC[(offset * num_of_algebraic) + tiF];
RATES[(offset * num_of_rates) + iS] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iS])/ALGEBRAIC[(offset * num_of_algebraic) + tiS];
RATES[(offset * num_of_rates) + iFp] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iFp])/ALGEBRAIC[(offset * num_of_algebraic) + tiFp];
RATES[(offset * num_of_rates) + iSp] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iSp])/ALGEBRAIC[(offset * num_of_algebraic) + tiSp];
RATES[(offset * num_of_rates) + Jrelnp] = (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - STATES[(offset * num_of_states) + Jrelnp])/ALGEBRAIC[(offset * num_of_algebraic) + tau_rel];
RATES[(offset * num_of_rates) + Jrelp] = (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - STATES[(offset * num_of_states) + Jrelp])/ALGEBRAIC[(offset * num_of_algebraic) + tau_relp];
RATES[(offset * num_of_rates) + ki] = ( - ((ALGEBRAIC[(offset * num_of_algebraic) + Ito]+ALGEBRAIC[(offset * num_of_algebraic) + IKr]+ALGEBRAIC[(offset * num_of_algebraic) + IKs]+ALGEBRAIC[(offset * num_of_algebraic) + IK1]+ALGEBRAIC[(offset * num_of_algebraic) + IKb]+ALGEBRAIC[(offset * num_of_algebraic) + Istim]) -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + JdiffK]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo];
RATES[(offset * num_of_rates) + kss] = ( - ALGEBRAIC[(offset * num_of_algebraic) + ICaK]*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + JdiffK];
RATES[(offset * num_of_rates) + nai] = ( - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab])*CONSTANTS[(offset * num_of_constants) + Acap]*CONSTANTS[(offset * num_of_constants) + cm])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo];
RATES[(offset * num_of_rates) + nass] = ( - (ALGEBRAIC[(offset * num_of_algebraic) + ICaNa]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa];
RATES[(offset * num_of_rates) + V] = - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ALGEBRAIC[(offset * num_of_algebraic) +  Ito]+ALGEBRAIC[(offset * num_of_algebraic) +  ICaL]+ALGEBRAIC[(offset * num_of_algebraic) +  ICaNa]+ALGEBRAIC[(offset * num_of_algebraic) + ICaK]+ALGEBRAIC[(offset * num_of_algebraic) + IKr]+ALGEBRAIC[(offset * num_of_algebraic) + IKs]+ALGEBRAIC[(offset * num_of_algebraic) + IK1]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss]+ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab]+ALGEBRAIC[(offset * num_of_algebraic) + IKb]+ALGEBRAIC[(offset * num_of_algebraic) + IpCa]+ALGEBRAIC[(offset * num_of_algebraic) + ICab]+ALGEBRAIC[(offset * num_of_algebraic) + Istim]);
RATES[(offset * num_of_rates) + cass] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcass]*((( - (ALGEBRAIC[(offset * num_of_algebraic) + ICaL] -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( 2.00000*CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss])+( ALGEBRAIC[(offset * num_of_algebraic) + Jrel]*CONSTANTS[(offset * num_of_constants) + vjsr])/CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + Jdiff]);
RATES[(offset * num_of_rates) + cai] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcai]*((( - ((ALGEBRAIC[(offset * num_of_algebraic) + IpCa]+ALGEBRAIC[(offset * num_of_algebraic) + ICab]) -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( 2.00000*CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo]) - ( ALGEBRAIC[(offset * num_of_algebraic) + Jup]*CONSTANTS[(offset * num_of_constants) + vnsr])/CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + Jdiff]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo]);
RATES[(offset * num_of_rates) + cansr] = ALGEBRAIC[(offset * num_of_algebraic) + Jup] - ( ALGEBRAIC[(offset * num_of_algebraic) + Jtr]*CONSTANTS[(offset * num_of_constants) + vjsr])/CONSTANTS[(offset * num_of_constants) + vnsr];
RATES[(offset * num_of_rates) + cajsr] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcajsr]*(ALGEBRAIC[(offset * num_of_algebraic) + Jtr] - ALGEBRAIC[(offset * num_of_algebraic) + Jrel]);
}

__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset)
{
  int num_of_constants = 146;
  int num_of_states = 41;
  int num_of_algebraic = 199;
  int num_of_rates = 41;
  
// #ifdef EULER
//   STATES[V] = STATES[V] + RATES[V] * dt;
//   STATES[CaMKt] = STATES[CaMKt] + RATES[CaMKt] * dt;
//   STATES[cass] = STATES[cass] + RATES[cass] * dt;
//   STATES[nai] = STATES[nai] + RATES[nai] * dt;
//   STATES[nass] = STATES[nass] + RATES[nass] * dt;
//   STATES[ki] = STATES[ki] + RATES[ki] * dt;
//   STATES[kss] = STATES[kss] + RATES[kss] * dt;
//   STATES[cansr] = STATES[cansr] + RATES[cansr] * dt;
//   STATES[cajsr] = STATES[cajsr] + RATES[cajsr] * dt;
//   STATES[cai] = STATES[cai] + RATES[cai] * dt;
//   STATES[m] = STATES[m] + RATES[m] * dt;
//   STATES[hf] = STATES[hf] + RATES[hf] * dt;
//   STATES[hs] = STATES[hs] + RATES[hs] * dt;
//   STATES[j] = STATES[j] + RATES[j] * dt;
//   STATES[hsp] = STATES[hsp] + RATES[hsp] * dt;
//   STATES[jp] = STATES[jp] + RATES[jp] * dt;
//   STATES[mL] = STATES[mL] + RATES[mL] * dt;
//   STATES[hL] = STATES[hL] + RATES[hL] * dt;
//   STATES[hLp] = STATES[hLp] + RATES[hLp] * dt;
//   STATES[a] = STATES[a] + RATES[a] * dt;
//   STATES[iF] = STATES[iF] + RATES[iF] * dt;
//   STATES[iS] = STATES[iS] + RATES[iS] * dt;
//   STATES[ap] = STATES[ap] + RATES[ap] * dt;
//   STATES[iFp] = STATES[iFp] + RATES[iFp] * dt;
//   STATES[iSp] = STATES[iSp] + RATES[iSp] * dt;
//   STATES[d] = STATES[d] + RATES[d] * dt;
//   STATES[ff] = STATES[ff] + RATES[ff] * dt;
//   STATES[fs] = STATES[fs] + RATES[fs] * dt;
//   STATES[fcaf] = STATES[fcaf] + RATES[fcaf] * dt;
//   STATES[fcas] = STATES[fcas] + RATES[fcas] * dt;
//   STATES[jca] = STATES[jca] + RATES[jca] * dt;
//   STATES[ffp] = STATES[ffp] + RATES[ffp] * dt;
//   STATES[fcafp] = STATES[fcafp] + RATES[fcafp] * dt;
//   STATES[nca] = STATES[nca] + RATES[nca] * dt;
//   STATES[xrf] = STATES[xrf] + RATES[xrf] * dt;
//   STATES[xrs] = STATES[xrs] + RATES[xrs] * dt;
//   STATES[xs1] = STATES[xs1] + RATES[xs1] * dt;
//   STATES[xs2] = STATES[xs2] + RATES[xs2] * dt;
//   STATES[xk1] = STATES[xk1] + RATES[xk1] * dt;
//   STATES[Jrelnp] = STATES[Jrelnp] + RATES[Jrelnp] * dt;
//   STATES[Jrelp] = STATES[Jrelp] + RATES[Jrelp] * dt;
// #else
////==============
  ////Exact solution
  ////==============
  ////INa
  STATES[(offset * num_of_states) + m] = ALGEBRAIC[(offset * num_of_algebraic) + mss] - (ALGEBRAIC[(offset * num_of_algebraic) + mss] - STATES[(offset * num_of_states) + m]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tm]);
  STATES[(offset * num_of_states) + hf] = ALGEBRAIC[(offset * num_of_algebraic) + hss] - (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + thf]);
  STATES[(offset * num_of_states) + hs] = ALGEBRAIC[(offset * num_of_algebraic) + hss] - (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ths]);
  STATES[(offset * num_of_states) + j] = ALGEBRAIC[(offset * num_of_algebraic) + jss] - (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + j]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tj]);
  STATES[(offset * num_of_states) + hsp] = ALGEBRAIC[(offset * num_of_algebraic) + hssp] - (ALGEBRAIC[(offset * num_of_algebraic) + hssp] - STATES[(offset * num_of_states) + hsp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + thsp]);
  STATES[(offset * num_of_states) + jp] = ALGEBRAIC[(offset * num_of_algebraic) + jss] - (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + jp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tjp]);
  STATES[(offset * num_of_states) + mL] = ALGEBRAIC[(offset * num_of_algebraic) + mLss] - (ALGEBRAIC[(offset * num_of_algebraic) + mLss] - STATES[(offset * num_of_states) + mL]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tmL]);
  STATES[(offset * num_of_states) + hL] = ALGEBRAIC[(offset * num_of_algebraic) + hLss] - (ALGEBRAIC[(offset * num_of_algebraic) + hLss] - STATES[(offset * num_of_states) + hL]) * exp(-dt / CONSTANTS[(offset * num_of_constants) + thL]);
  STATES[(offset * num_of_states) + hLp] = ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - (ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - STATES[(offset * num_of_states) + hLp]) * exp(-dt / CONSTANTS[(offset * num_of_constants) + thLp]);
  ////Ito
  STATES[(offset * num_of_states) + a] = ALGEBRAIC[(offset * num_of_algebraic) + ass] - (ALGEBRAIC[(offset * num_of_algebraic) + ass] - STATES[(offset * num_of_states) + a]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ta]);
  STATES[(offset * num_of_states) + iF] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iF]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiF]);
  STATES[(offset * num_of_states) + iS] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iS]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiS]);
  STATES[(offset * num_of_states) + ap] = ALGEBRAIC[(offset * num_of_algebraic) + assp] - (ALGEBRAIC[(offset * num_of_algebraic) + assp] - STATES[(offset * num_of_states) + ap]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ta]);
  STATES[(offset * num_of_states) + iFp] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iFp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiFp]);
  STATES[(offset * num_of_states) + iSp] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iSp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiSp]);
  ////ICaL
  STATES[(offset * num_of_states) + d] = ALGEBRAIC[(offset * num_of_algebraic) + dss] - (ALGEBRAIC[(offset * num_of_algebraic) + dss] - STATES[(offset * num_of_states) + d]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + td]);
  STATES[(offset * num_of_states) + ff] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ff]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tff]);
  STATES[(offset * num_of_states) + fs] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + fs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfs]);
  STATES[(offset * num_of_states) + fcaf] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcaf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfcaf]);
  STATES[(offset * num_of_states) + fcas] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcas]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfcas]);
  STATES[(offset * num_of_states) + jca] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + jca]) * exp(- dt / CONSTANTS[(offset * num_of_constants) + tjca]);
  STATES[(offset * num_of_states) + ffp] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ffp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tffp]);
  STATES[(offset * num_of_states) + fcafp] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcafp]) * exp(-d / ALGEBRAIC[(offset * num_of_algebraic) + tfcafp]);
  STATES[(offset * num_of_states) + nca] = ALGEBRAIC[(offset * num_of_algebraic) + anca] * CONSTANTS[(offset * num_of_constants) + k2n] / ALGEBRAIC[(offset * num_of_algebraic) + km2n] - (ALGEBRAIC[(offset * num_of_algebraic) + anca] * CONSTANTS[(offset * num_of_constants) + k2n] / ALGEBRAIC[(offset * num_of_algebraic) + km2n] - STATES[(offset * num_of_states) + nca]) * exp(-ALGEBRAIC[(offset * num_of_algebraic) + km2n] * dt);
  ////IKr
  STATES[(offset * num_of_states) + xrf] = ALGEBRAIC[(offset * num_of_algebraic) + xrss] - (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txrf]);
  STATES[(offset * num_of_states) + xrs] = ALGEBRAIC[(offset * num_of_algebraic) + xrss] - (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txrs]);
  ////IKs
  STATES[(offset * num_of_states) + xs1] = ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - STATES[(offset * num_of_states) + xs1]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txs1]);
  STATES[(offset * num_of_states) + xs2] = ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - STATES[(offset * num_of_states) + xs2]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txs2]);
  ////IK1
  STATES[(offset * num_of_states) + xk1] = ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - STATES[(offset * num_of_states) + xk1]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txk1]);
  ////INaCa
  ////INaK
  ////IKb
  ////INab
  ////ICab
  ///IpCa
  ////Diffusion fluxes
  ////RyR receptors
  STATES[(offset * num_of_states) + Jrelnp] = ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - STATES[(offset * num_of_states) + Jrelnp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tau_rel]);
  STATES[(offset * num_of_states) + Jrelp] = ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - STATES[(offset * num_of_states) + Jrelp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tau_relp]);
  ////SERCA Pump
  ////Calcium translocation
  //
  ////=============================
  ////Approximated solution (Euler)
  ////=============================
  ////ICaL
  //STATES[jca] = STATES[jca] + RATES[jca] * dt;
  ////CaMK
  STATES[(offset * num_of_states) + CaMKt] = STATES[(offset * num_of_states) + CaMKt] + RATES[(offset * num_of_rates) + CaMKt] * dt;
  ////Membrane potential
  STATES[(offset * num_of_states) + V] = STATES[(offset * num_of_states) + V] + RATES[(offset * num_of_rates) + V] * dt;
  ////Ion Concentrations and Buffers
  STATES[(offset * num_of_states) + nai] = STATES[(offset * num_of_states) + nai] + RATES[(offset * num_of_rates) + nai] * dt;
  STATES[(offset * num_of_states) + nass] = STATES[(offset * num_of_states) + nass] + RATES[(offset * num_of_rates) + nass] * dt;
  STATES[(offset * num_of_states) + ki] = STATES[(offset * num_of_states) + ki] + RATES[(offset * num_of_rates) + ki] * dt;
  STATES[(offset * num_of_states) + kss] = STATES[(offset * num_of_states) + kss] + RATES[(offset * num_of_rates) + kss] * dt;
  STATES[(offset * num_of_states) + cai] = STATES[(offset * num_of_states) + cai] + RATES[(offset * num_of_rates) + cai] * dt;
  STATES[(offset * num_of_states) + cass] = STATES[(offset * num_of_states) + cass] + RATES[(offset * num_of_rates) + cass] * dt;
  STATES[(offset * num_of_states) + cansr] = STATES[(offset * num_of_states) + cansr] + RATES[(offset * num_of_rates) + cansr] * dt;
  STATES[(offset * num_of_states) + cajsr] = STATES[(offset * num_of_states) + cajsr] + RATES[(offset * num_of_rates) + cajsr] * dt; 
// #endif
}

__device__ double set_time_step(double TIME,
  double time_point,
  double max_time_step,
  double *CONSTANTS,
  double *RATES,
  double *STATES,
  double *ALGEBRAIC,
  int offset) {
  double time_step = 0.005;
  int num_of_constants = 146;
  int num_of_rates = 41;

  if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[BCL + (offset * num_of_constants)]) * CONSTANTS[BCL + (offset * num_of_constants)]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("dV = %lf, time_step = %lf\n",RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) <= 0.2) {//Slow changes in V
        // printf("dV/dt <= 0.2\n");
        time_step = std::abs(0.8 / RATES[V + (offset * num_of_rates)]);
        //Make sure time_step is between 0.005 and max_time_step
        if (time_step < 0.005) {
            time_step = 0.005;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
    }
    else if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) >= 0.8) {//Fast changes in V
        // printf("dV/dt >= 0.8\n");
        time_step = std::abs(0.2 / RATES[V + (offset * num_of_rates)]);
        while (std::abs(RATES[V + (offset * num_of_rates)]  * time_step) >= 0.8 &&
               0.005 < time_step &&
               time_step < max_time_step) {
            time_step = time_step / 10.0;
            // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
        }
    }
    // __syncthreads();
    return time_step;
  }
}

