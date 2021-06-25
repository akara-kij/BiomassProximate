%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% # Released under MIT License %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2021 Akara Kijkarncharoensin, akara_kij@utcc.ac.th        %
% Department of Computer Engineering and Financial Technology,            %
% School of Engineering, University of the Thai Chamber of Commerce.      %
%                                                                         %
% Permission is hereby granted, free of charge, to any person obtaining a %
% copy of this software and associated documentation files (the           %
% "Software") , to deal in the Software without restriction, including    %
% without limitation the rights to use, copy, modify, merge, publish,     %
% distribute, sublicense, and/or sell copies of the Software , and to     %
% permit persons to whom the Software is furnished to do so, subject to   %
% the following conditions:                                               %
%                                                                         %
% The above copyright notice and this permission notice shall be included %
% in all copies or substantial portions of the Software.                  %
%                                                                         %
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,         %
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF      %
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  %
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    %
% CLAIM,DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT% 
% OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR%
% THE USE OR OTHER DEALINGS IN THE SOFTWARE.                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
% Pre-Opening %
%%%%%%%%%%%%%%%
clc;
clear;
close all;
format shortEng;

% Program parameter
feature = ["FC" "VM" "ASH" "HHV"];
ResponseVarName = "HHV";
strGroup = {"This study", "H.Qian","S.B.Ghugare","D.R.Nhuchhen"};
strMarker  = {"+",'o','^','d','p'};
numAlpha = [ 1.0,0.8,0.8,0.8 ];

% Load research database
filename = "ThisStudy";
research = fitnetRL(feature,ResponseVarName);
research.load("\BiomassDataSetProximate.xlsx",filename);
TBL = research.dataTBL;
TBL.Grp = repmat( strGroup{1} , length(TBL.Variables),1);
TBL.Grp = categorical( TBL.Grp );
BioData = TBL;

% Load Qian (2016) database
filename = "Qian2016";
Qian= fitnetRL(feature,ResponseVarName);
Qian.load("\BiomassDataSetProximate.xlsx",filename);
TBL = Qian.dataTBL;
TBL.Grp = repmat( strGroup{2} , length(TBL.Variables),1);
TBL.Grp = categorical( TBL.Grp );
BioData = [BioData ; TBL];

% Load Ghugare (2017)  table S1
filename = "GhugareS1";
Ghugare = fitnetRL(feature,ResponseVarName);
Ghugare.load("\BiomassDataSetProximate.xlsx",filename);
TBL = Ghugare.dataTBL;
TBL.Grp = repmat( strGroup{3} , length(TBL.Variables),1);
TBL.Grp = categorical( TBL.Grp );
BioData = [BioData ; TBL];

% Load Nhuchhen (2017)  table S2
filename = "NhuchhenS2";
Nhuchhen = fitnetRL(feature,ResponseVarName);
Nhuchhen.load("\BiomassDataSetProximate.xlsx",filename);
TBL = Nhuchhen.dataTBL;
TBL.Grp = repmat( strGroup{4} , length(TBL.Variables),1);
TBL.Grp = categorical( TBL.Grp );
BioData = [BioData ; TBL];

% Display the univser of biomass raw data in each database
figName = "Biomass raw data in the literature";
fig1 = figure("Name",figName);
for i = 1:length(strGroup)
    scatter3(BioData{BioData.Grp==strGroup{i},1},BioData{BioData.Grp==strGroup{i},2},BioData{BioData.Grp==strGroup{i},3},strMarker{i},'LineWidth',1,'MarkerEdgeAlpha',numAlpha(i));
    hold on
end 
hold off
grid on;
grid minor;
set(gca,'FontName','Times New Roman');
title('Database of Biomass Raw Data','interpreter','latex');
xlabel(feature{1},'interpreter','latex');
ylabel(feature{2},'interpreter','latex');
zlabel(feature{3},'interpreter','latex');
xlim([0 100]);
ylim([0 100]);
zlim([0 100]);
view(gca,[-160 15]);
legend( strGroup ,'location','best','interpreter','latex');

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the pridicted model %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
% Routine parameter
feature         = ["FC" "VM" "ASH" "HHV"];
ResponseVarName = "HHV";
FileName        = "\BiomassDataSetProximate.xlsx";
WorkSheet       = "ThisStudy";
NFold           = 20;
lambda          = linspace(0.01,2,1000);

%%%%%%%%%%%%%%%%%%%%%%
% Neural Network OOP %
%%%%%%%%%%%%%%%%%%%%%%
% Implement model ANN331 from Harun Uzun et al. 2017
nNeuron         = [3];
trainFcn        = 'trainlm';
transFcn        = {'tansig' };
bioNN           = fitnetRL(feature,ResponseVarName);
bioNN.load(FileName,WorkSheet);
NAll            = length(bioNN.dataTBL{:,:});
c               = cvpartition(NAll,'KFold',NFold);
bioNN.fit('nNeuron',nNeuron,'trainFcn',trainFcn,'transFcn',transFcn,'cvpartition',c,'showResources','yes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Radial Basis Netwrok OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default value of maixmum number of neurons (MN) equal to the number of data
% Cannot repeat Amir Dashti 2019 et. al that use MN = 27
bioRB = newrbRL(feature,ResponseVarName);
bioRB.load(FileName,WorkSheet);
bioRB.fit('cvpartition',c,'showResources','yes','spread',0.5722,'MN',27);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adaptive Neuro Fuzzy Inference System ( ANFIS ) OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement mdoel ANFIS-SC5 from Ebru Akkaya 2016
bioFIS = genfisRL(feature,ResponseVarName);
bioFIS.load(FileName,WorkSheet);
bioFIS.tuneParameter("Method","anfis", ...
                     "OptimizationType","learning", ...
                     "DistanceMetric","rmse",...
                     "NumMaxRules",300);

bioFIS.fit('cvpartition',c, ...
           'clusteringType','SubtractiveClustering', ...
           'ClusterInfluenceRange',0.14,...
           'SquashFactor',1.4, ...
           'RejectRatio',0.15, ...
           'AcceptRatio',0.5,...
           'Verbos',true);
bioFIS.mdlDiagnostic(bioFIS.mdl.fold1);
showrule(bioFIS.mdl.fold1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stepwise linear regression %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement MATLAB stepwise and robust regression
bioLM = fitlmRL(feature,ResponseVarName);
bioLM.load(FileName,WorkSheet);
bioLM.fit('cvpartition',c,'Upper','quadratic','Verbose',2, 'PEnter',0.05,'Standardize',true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supported Vector Machine OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement MATLAB support vector machine with all hyperparameter optimization
bioSVM = fitsvmRL(feature,ResponseVarName);
bioSVM.load(FileName,WorkSheet);
bioSVM.fit('cvpartition',c,'Standardize',true,'OptimizeHyperparameters',"all");

%%%%%%%%%%%%%%%%%%%%%%%
% Regression tree OOP %
%%%%%%%%%%%%%%%%%%%%%%%
% Implement the regress tree model with all hyperparameter optimization
bioTree = fittreeRL(feature,ResponseVarName);
bioTree.load(FileName,WorkSheet);
bioTree.fit('cvpartition',c,'OptimizeHyperparameters',"all");

%%%%%%%%%%%%%%%%%%%%%%
% Ensemble tree OOP  %
%%%%%%%%%%%%%%%%%%%%%%
% Implement the ensemble regress tree model with all hyperparameter optimization
bioENS = fitensembleRL(feature,ResponseVarName);
bioENS.load(FileName,WorkSheet);
bioENS.fit('cvpartition',c,'OptimizeHyperparameters',"all");

%%%%%%%%%%%%%%%%%%%%%%%%
% Guassian Process OOP %
%%%%%%%%%%%%%%%%%%%%%%%%
% Implement the Gaussian process model with all hyperparameter optimization
bioGP = fitgpRL(feature,ResponseVarName);
bioGP.load(FileName,WorkSheet);
bioGP.fit('cvpartition',c,'OptimizeHyperparameters',"all");

%%%%%%%%%%%%%%%%%%%%%%%%
% Ridge Regression OOP %
%%%%%%%%%%%%%%%%%%%%%%%%
% Implement ridge regression with predictor in quardatic form
bioRD  = ridgeRL(feature,ResponseVarName);
bioRD.load(FileName,WorkSheet);
bioRD.fit('cvpartition',c,'Lambda',lambda,'Scaled',0,'model','Quadratic');

%%%%%%%%%%%%%%%%%%%%%%%%
% Lasso Regression OOP %
%%%%%%%%%%%%%%%%%%%%%%%%
% Implement lasso regression with predictor in quardatic form
bioLAS = lassoRL(feature,ResponseVarName);
bioLAS.load(FileName,WorkSheet);
bioLAS.fit('cvpartition',c,'Lambda',lambda,'model','Quadratic','Alpha',1.0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generalized Linear Model OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement generalize linear model 
strFormula      = "HHV ~ 1 + FC*ASH + FC^2";
bioGLM          = fitglmRL(feature,ResponseVarName);
bioGLM.Formula  = strFormula;
bioGLM.load(FileName,WorkSheet);
bioGLM.fit('cvpartition',c,'Distribution','normal');

%%%%%%%%%%%%%%%%%
% Lasso GLM OOP %
%%%%%%%%%%%%%%%%%
% Implement lasso generalized linear model 
bioLGLM = lassoglmRL(feature,ResponseVarName);
bioLGLM.load(FileName,WorkSheet);
bioLGLM.fit('cvpartition',c,'Link','identity','Distribution','normal','Lambda',lambda);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partial least square OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bioPLS = plsregRL(feature,ResponseVarName);
bioPLS.load(FileName,WorkSheet);
bioPLS.fit('cvpartition',c,'Lambda',lambda);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Higher-Dimensional Linear Model OOP %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bioHD = fitrlinearRL(feature,ResponseVarName);
bioHD.load(FileName,WorkSheet);
bioHD.fit('cvpartition',c,'Lambda',lambda);
time = toc;
fprintf("Computational Time : %.2f \n", time/60.);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The report of model performance %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section parameters
rowNames = {'Stepwise'  , ...
            'Robust Lin', ...
            'HD Lin'    , ...
            'GLM'       , ...
            'PLS'       , ...
            'Ridge'     , ...
            'Lasso'     , ...
            'Lasso GLM' , ...
            'ANFIS'     , ...
            'MLP-ANN'   , ...
            'SVM'       , ...
            'GP'        , ...
            'Ensemble'  , ...
            'Tree'      , ...
            'RBF'       };
colNames = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};      
        
errData  = [bioLM.EvaTBL.Step.CrossVal.CrossVal     , ...
            bioLM.EvaTBL.Robust.CrossVal.CrossVal   , ...
            bioHD.EvaTBL.CrossVal.CrossVal          , ...
            bioGLM.EvaTBL.CrossVal.CrossVal         , ...
            bioPLS.EvaTBL.CrossVal.CrossVal         , ...
            bioRD.EvaTBL.CrossVal.CrossVal          , ...
            bioLAS.EvaTBL.CrossVal.CrossVal         , ...
            bioLGLM.EvaTBL.CrossVal.CrossVal        , ...
            bioFIS.EvaTBL.CrossVal.CrossVal         , ...
            bioNN.EvaTBL.CrossVal.CrossVal          , ...
            bioSVM.EvaTBL.CrossVal.CrossVal         , ...
            bioGP.EvaTBL.CrossVal.CrossVal          , ...
            bioENS.EvaTBL.CrossVal.CrossVal         , ...
            bioTree.EvaTBL.CrossVal.CrossVal        , ...
            bioRB.EvaTBL.CrossVal.CrossVal          ]';
        
% Create the report table        
errTBL =   table( errData(:,1),errData(:,2),errData(:,3),errData(:,4),errData(:,5),errData(:,6),errData(:,7) , 'VariableNames', colNames, 'RowNames', rowNames );   

fprintf("The Model Performance\n");
fprintf("=====================\n");
disp(errTBL)           

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manually create the data of residuals %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The residuals of all models : Collect data manually
residualsDB   = [   bioLM.Residuals.Step.CrossVal.Data  ; ...   % Stepwise linear regression
                    bioLM.Residuals.Robust.CrossVal.Data; ...   % Robust linear regression
                    bioHD.Residuals.CrossVal.Data       ; ...   % High-dimensional data linear regression
                    bioGLM.Residuals.CrossVal.Data      ; ...   % Generalize linear regression model
                    bioPLS.Residuals.CrossVal.Data      ; ...   % Partial lear-squares regression
                    bioRD.Residuals.CrossVal.Data       ; ...   % Ridge regression
                    bioLAS.Residuals.CrossVal.Data      ; ...   % Lasso or elastic net regularization for linear models
                    bioLGLM.Residuals.CrossVal.Data     ; ...   % Lasso or elastic net regularization for generalized linear models 
                    bioFIS.Residuals.CrossVal.Data      ; ...   % Generate fuzzy inference system objiect from data
                    bioNN.Residuals.CrossVal.Data       ; ...   % Neural network function
                    bioSVM.Residuals.CrossVal.Data      ; ...   % Support vector machine regression model
                    bioGP.Residuals.CrossVal.Data       ; ...   % Gaussian process regression model
                    bioENS.Residuals.CrossVal.Data      ; ...   % Fit ensemble of learners for regression
                    bioTree.Residuals.CrossVal.Data     ; ...   % Binary decision tree for regression  
                    bioRB.Residuals.CrossVal.Data       ];      % Radial basis network

residualsTBL  = table( residualsDB, MdlName,'VariableNames',{'Data','Name'} );   


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Effect of database to the prediction performance %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
% Load the backup subresidual data
rawData    = readtable("BiomassDataSetProximate.xlsx","FileType","spreadsheet","Sheet","Full Residuals");

% Statistic table of the residuals of the prediction model
statarray = grpstats(rawData,{'Database'},{'mean',@(x)median(x),@(x)skewness(x),@(x)kurtosis(x),'sem','meanci'},'DataVars','Data','Varnames',{'Database','Count','Mean','Median','Skewness','Kurtosis','SE','C.I.'});
fprintf("Descriptive statistic of the residuals\n");
fprintf("======================================\n");
disp(statarray);

% Indicate the distribution of the residuals received from the database.
figure('name','Dataset equivalent of the testing model')
strTitle = 'Model residuals given on 13 models';
%boxplot(rawData.Data,rawData.Database,'DataLim',[-15,15],'ExtremeMode','compress');
boxplot(rawData.Data,rawData.Database,'DataLim',[-15,15],'ExtremeMode','compress');
title( strTitle,'interpreter','latex');
xlabel('Dataset','interpreter','latex');
ylabel('Model Residuals','interpreter','latex');
ylim([-10 10]);
grid on
grid minor;