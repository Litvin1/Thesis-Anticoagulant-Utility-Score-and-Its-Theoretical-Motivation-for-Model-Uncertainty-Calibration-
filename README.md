Thesis-Anticoagulant-Utility-Score-and-Its-Theoretical-Motivation-for-Model-Uncertainty-Calibration-

In this thesis, we tried to build a venous thromboembolism (VTE) prediction tool that was better than the Pauda score, which is used now in all hospitals in the state. We outperformed the Pauda score and showed statistical significance to those results. Later on, we tried to address the problem of major bleeding that can occur after treatment with anticoagulants for preventing VTE. To solve this, we defined the Anti-Coagulant Utility (ACU) score. To show the statistical power of ACU, we used confidence intervals for a population mean. For choosing the algorithm to build the submodels that construct the ACU, we tried to give a theoretical motivation for choosing algorithms that will produce models with good uncertainty calibration. The proofs are in appendices A and B.

PDF file of the thesis table of contents:
Pages 1–9: introduction. biological background: What is Venous Thrombo-Embolism (VTE)? What is the Padua score? What is major bleeding? Algorithmic background: what is machine learning? What is risk and Empirical Risk Minimization (ERM)? How does Hoeffding's inequality help us? describing the datasets.
Pages 10–18: Methods. Python language and libreries, Anti-Coagulant Utility (ACU) score definition, feature selection, missing values, performance evaluation; Statistical tests: paired sample t-test, TOST (two one-sided t-tests) equivalence test, confidence intervals for a population mean, model uncertainty calibration, theoretical claims, algorithm
Pages 19–25: Results. predicting VTE with new weights for Padua variables, predicting VTE with new variables, predicting major bleeding, confidence intervals for ACU between different populations.
Pages 26–28: Discussion
Pages 29–32: Bibliography
Pages 33–37: Appendix A. Proof for the claim: if both models for predicting VTE and major bleeding are perfectly calibrated, then the conditional expectation of the ACU is higher for the population of patients who had a VTE event than the major bleeding population.
Pages 34–38: Appendix B. Proof for the claim: if both models for predicting VTE and major bleeding are perfectly calibrated, then the average ACU of a final size sample of patients who had a VTE event is higher than the average ACU of the major bleeding final sample.
