# References

Every entry below was checked by web search on 2026-10-28 (title, authors, year and a
resolving URL or DOI). Where only the abstract or a bibliographic record could be checked,
this is stated. Entries are grouped by the role they play in RobustPrint. If a statement
in this repository is not backed by one of these entries, it is labelled as an
**assumption** in [`DECISIONS.md`](DECISIONS.md).

## In-space manufacturing and recycling (framing only)

- **[NASA-REFAB]** NASA Tech Briefs. "NASA's ReFabricator Recycles Waste Plastic into New
  Parts." <https://www.techbriefs.com/component/content/article/37861-nasa-s-refabricator-recycles-waste-plastic-into-new-parts>.
  Also: GeekWire (2018), "Tethers Unlimited delivers 3-D printer/recycler combo for NASA's
  space station", <https://www.geekwire.com/2018/tethers-unlimited-delivers-3-d-printer-recycler-combo-nasa-space-station/>.
  Used for: the Refabricator (Tethers Unlimited, built for NASA MSFC under SBIR) combines
  recycling and FDM printing and was tested on the ISS; the recycling process was
  reported as "Positrusion".
- **[PRATER-2016]** Prater, T. J., Bean, Q. A., et al. (2016). *Summary Report on Phase I
  Results from the 3D Printing in Zero G Technology Demonstration Mission, Volume I.*
  NASA/TP-2016-219101. <https://ntrs.nasa.gov/citations/20160008972>. Used for: the first
  FDM printer on the ISS (installed November 2014). Only the record and summaries were read.
- **[BLANCHARD-NTRS]** NASA Technical Reports Server document 20210023437 (Blanchard, A. J.).
  <https://ntrs.nasa.gov/api/citations/20210023437/downloads/BlanchardAlexanderJ.pdf>.
  Used for: packaging (foam and films) dominates ISS logistics waste. Only the search
  excerpt was read; we do not quote numbers from it.

## FDM process physics

- **[SUN-2008]** Sun, Q., Rizvi, G. M., Bellehumeur, C. T., Gu, P. (2008). Effect of
  processing conditions on the bonding quality of FDM polymer filaments. *Rapid
  Prototyping Journal* 14(2), 72–80. <https://doi.org/10.1108/13552540810862028>.
  Used for: inter-filament bonding is governed by the time the interface stays above a
  critical temperature; envelope temperature and convection (fan) affect it strongly.
- **[WOOL-1981]** Wool, R. P., O'Connor, K. M. (1981). A theory of crack healing in polymers.
  *Journal of Applied Physics* 52, 5953. Citation verified through secondary sources
  (e.g. <https://www.ideals.illinois.edu/items/71978>). Used for: interface strength grows
  with contact time roughly as t^(1/4) before full healing (reptation).
- **[AHN-2002]** Ahn, S.-H., Montero, M., Odell, D., Roundy, S., Wright, P. K. (2002).
  Anisotropic material properties of fused deposition modeling ABS. *Rapid Prototyping
  Journal* 8(4), 248–257. <https://doi.org/10.1108/13552540210441166>. Used for: raster
  orientation dominates FDM tensile strength; FDM tensile strength was about 65–72 % of
  injection-moulded ABS.
- **[CUAN-2019]** Cuan-Urquizo, E., et al. (2019). Characterization of the mechanical
  properties of FFF structures and materials: a review on the experimental, computational
  and theoretical approaches. *Materials* 12(6), 895. <https://doi.org/10.3390/ma12060895>.
  Used for: general background on FFF mechanical-property modelling.

## Materials and mixing models

- **[ETB-MODULUS]** The Engineering ToolBox. "Young's Modulus, Tensile Strength and Yield
  Strength Values for some Materials." <https://www.engineeringtoolbox.com/young-modulus-d_417.html>
  (accessed 2026-10-28). Reference database, not peer reviewed. Used for: bulk modulus and
  strength ranges of HDPE (0.8 GPa, 15 MPa), PP (1.5–2 GPa, 28–36 MPa), PET (2–2.7 GPa,
  55 MPa), nylon-6 (2–4 GPa, 45–90 MPa), ABS (1.4–3.1 GPa, 40 MPa), aramid (70–112 GPa).
- **[ETB-POLYMER]** The Engineering ToolBox. "Polymers – Physical Properties."
  <https://www.engineeringtoolbox.com/polymer-properties-d_1222.html> (accessed 2026-10-28).
  Used for: polyethylene density about 950 kg/m³, tensile strength 20–30 MPa, modulus 0.7 GPa;
  nylon density 1160 kg/m³.
- **[FARAH-2016]** Farah, S., Anderson, D. G., Langer, R. (2016). Physical and mechanical
  properties of PLA, and their functions in widespread applications — a comprehensive
  review. *Advanced Drug Delivery Reviews* 107, 367–392. <http://hdl.handle.net/1721.1/112940>.
  Used for: PLA immiscibility with other polymers and thermal degradation as general context
  (abstract-level only).
- **[HALPIN-1976]** Halpin, J. C., Kardos, J. L. (1976). The Halpin–Tsai equations: a review.
  *Polymer Engineering and Science* 16(5), 344–352. <https://doi.org/10.1002/pen.760160512>.
  Used for: the Halpin–Tsai modulus model for short-fibre composites.
- **[NICOLAIS-1970]** Nicolais, L., Narkis, M. (1970). *Studies of Stress-Strain Behavior of
  SAN/Glass Bead Composites in the Glassy Region.* Technical report, Washington University
  (DTIC AD0876335). <https://apps.dtic.mil/sti/html/tr/AD0876335/index.html>. Used for: the
  strength of polymers with non-adhering inclusions falls with filler volume fraction. The
  specific form σ_c = σ_m (1 − 1.21 φ^(2/3)) is commonly attributed to these authors; we
  could not check the equation in the report, so we treat the coefficient as an assumption.
- **[PPPE-BLENDS]** Open-access RSC Applied Polymers (2026) article on post-consumer
  recycled PP/PE blends, <https://pubs.rsc.org/en/content/articlelanding/2026/lp/d5lp00321k>,
  and the University of Leoben record "Effect of compatibilizers on blends of immiscible
  polymers containing polyolefins". Used for: immiscible polyolefin blends from mixed
  recycling show poor phase adhesion and reduced mechanical properties. Abstract-level only.

## Statistics, uncertainty and machine learning

- **[EGOZCUE-2003]** Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., Barceló-Vidal, C.
  (2003). Isometric logratio transformations for compositional data analysis. *Mathematical
  Geology* 35(3), 279–300. <https://doi.org/10.1023/A:1023818214614>.
- **[RW-2006]** Rasmussen, C. E., Williams, C. K. I. (2006). *Gaussian Processes for Machine
  Learning.* MIT Press. <https://gaussianprocess.org/gpml/>.
- **[KERSTING-2007]** Kersting, K., Plagemann, C., Pfaff, P., Burgard, W. (2007). Most likely
  heteroscedastic Gaussian process regression. *ICML 2007*, 393–400.
  <https://doi.org/10.1145/1273496.1273546>.
- **[LAKSH-2017]** Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable
  predictive uncertainty estimation using deep ensembles. *NeurIPS 30*.
  <https://arxiv.org/abs/1612.01474>.
- **[GAL-2016]** Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
  representing model uncertainty in deep learning. <https://arxiv.org/abs/1506.02142>.
- **[LEI-2018]** Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018).
  Distribution-free predictive inference for regression. *JASA* 113, 1094–1111.
  <https://doi.org/10.1080/01621459.2017.1307116>.
- **[ROMANO-2019]** Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile
  regression. *NeurIPS 32*. <https://arxiv.org/abs/1905.03222>.
- **[KULESHOV-2018]** Kuleshov, V., Fenner, N., Ermon, S. (2018). Accurate uncertainties for
  deep learning using calibrated regression. *ICML 2018*, PMLR 80, 2796–2804.
  <https://proceedings.mlr.press/v80/kuleshov18a.html>.
- **[GNEITING-2007]** Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA* 102(477), 359–378.
  <https://doi.org/10.1198/016214506000001437>.
- **[JULIER-2004]** Julier, S. J., Uhlmann, J. K. (2004). Unscented filtering and nonlinear
  estimation. *Proceedings of the IEEE* 92(3), 401–422.
  <https://doi.org/10.1109/JPROC.2003.823141>.
- **[SOBOL-2001]** Sobol′, I. M. (2001). Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates. *Mathematics and Computers in Simulation* 55,
  271–280.
- **[SALTELLI-2010]** Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M.,
  Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and
  estimator for the total sensitivity index. *Computer Physics Communications* 181(2),
  259–270. <https://publications.jrc.ec.europa.eu/repository/handle/JRC52955>.
- **[ROCKAFELLAR-2000]** Rockafellar, R. T., Uryasev, S. (2000). Optimization of conditional
  value-at-risk. *Journal of Risk* 2(3), 21–41.
  <https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk>.
- **[CAKMAK-2020]** Cakmak, S., Astudillo, R., Frazier, P., Zhou, E. (2020). Bayesian
  optimization of risk measures. *NeurIPS 33*. <https://arxiv.org/abs/2007.05554>.
- **[BALANDAT-2020]** Balandat, M., et al. (2020). BoTorch: a framework for efficient
  Monte-Carlo Bayesian optimization. *NeurIPS 33*.
  <https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html>.
- **[HANSEN-2016]** Hansen, N. (2016). The CMA evolution strategy: a tutorial.
  <https://arxiv.org/abs/1604.00772>.
- **[HOULSBY-2011]** Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. (2011). Bayesian
  active learning for classification and preference learning. <https://arxiv.org/abs/1112.5745>.
- **[HOLM-1979]** Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics* 6(2), 65–70. <https://www.jstor.org/stable/4615733>.

## Datasets

- **[KAGGLE-3DP]** Okudan, A. "3D Printer Dataset for Mechanical Engineers." Kaggle,
  <https://www.kaggle.com/datasets/afumetto/3dprinter> (version 3, last updated 2018-09-14).
  50 rows; 9 settings (layer height, wall thickness, infill density, infill pattern, nozzle
  temperature, bed temperature, print speed, material PLA/ABS, fan speed) and 3 measured
  outputs (roughness, tensile strength, elongation). Collected at Selçuk University on an
  Ultimaker S5 according to the dataset description. **Licence: listed as "Unknown" by
  Kaggle.** We therefore do not redistribute it; `make public-data` downloads it to the
  git-ignored `data/external/` folder and the benchmark that uses it is optional.
