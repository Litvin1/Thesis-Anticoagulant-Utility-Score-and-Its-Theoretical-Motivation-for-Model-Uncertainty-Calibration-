Thesis-Anticoagulant-Utility-Score-and-Its-Theoretical-Motivation-for-Model-Uncertainty-Calibration-

In this work, we tried to build a venous thromboembolism (VTE) prediction tool that was better than the Pauda score, which is used now in all hospitals in the state. We outperformed the Pauda score and showed statistical significance to those results. Later on, we tried to address the problem of major bleeding that can occur after treatment with anticoagulants for preventing VTE. To solve this, we defined the Anti-Coagulant Utility (ACU) score. To show the statistical significance of ACU, we used confidence intervals for a population mean (alpha = 5%).<br />
![image](https://user-images.githubusercontent.com/59581554/232444048-3ee4986d-950f-414b-8ea9-7c3ba14a7c63.png)<br />
![image](https://user-images.githubusercontent.com/59581554/232444084-baca5008-8a8a-40d0-a8e5-b193decfff9a.png)<br />
![image](https://user-images.githubusercontent.com/59581554/232444106-64c48364-e0b9-40b3-a7d1-08507666ba4e.png)<br />
![image](https://user-images.githubusercontent.com/59581554/232444132-e39e00f7-2b34-4bf8-8c01-11304523b8bb.png)<br />
The x axis is the ACU score, the y axis is the fold number, and the different plots are from different K-fold cross validation (5, 4, 3, 2). The colors blue, red and green represent the VTE, Bleeding, and Neither populations respectively. <br />
For choosing the algorithm to build the submodels that construct the ACU, we tried to give a theoretical motivation for choosing algorithms that will produce models with good uncertainty calibration. The proofs are in appendices A and B.<br />
The PDF file: <br />
Pages 1–9: introduction. biological background: What is Venous Thrombo-Embolism (VTE)? What is the Padua score? What is major bleeding? Algorithmic background: what is machine learning? What is risk and Empirical Risk Minimization (ERM)? How does Hoeffding's inequality help us? describing the datasets.<br />
Pages 10–18: Methods. Python language and libreries, Anti-Coagulant Utility (ACU) score definition, feature selection, missing values, performance evaluation; Statistical tests: paired sample t-test, TOST (two one-sided t-tests) equivalence test, confidence intervals for a population mean, model uncertainty calibration, theoretical claims, algorithm<br />
Pages 19–25: Results. predicting VTE with new weights for Padua variables, predicting VTE with new variables, predicting major bleeding, confidence intervals for ACU between different populations.<br />
Pages 26–28: Discussion<br />
Pages 29–32: Bibliography<br />
Pages 33–37: Appendix A. Proof for the claim: if both models for predicting VTE and major bleeding are perfectly calibrated, then the conditional expectation of the ACU is higher for the population of patients who had a VTE event than the major bleeding population.<br />
Pages 34–38: Appendix B. Proof for the claim: if both models for predicting VTE and major bleeding are perfectly calibrated, then the average ACU of a final size sample of patients who had a VTE event is higher than the average ACU of the major bleeding final sample.<br />
