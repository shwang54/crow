% load tadpole
tadpole = readtable('TADPOLE_D1_D2.csv');

%%
% find indices of samples with AV45
av45_idx = ~isnan(tadpole.AV45);


column_names = {'RID', ...
                'AGE', ...
                'DX_bl', ...
                'DXCHANGE', ...
                'CDRSB', ...
                'ADAS11', ...
                'ADAS13', ...
                'MMSE', ...
                'RAVLT_immediate', ...
                'RAVLT_learning', ...
                'RAVLT_forgetting', ...
                'FAQ', ...
                'MOCA', ...
                'Years_bl', ...
                'CTX_LH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_FRONTALPOLE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_FUSIFORM_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_INSULA_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_LINGUAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PARACENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PARSORBITALIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PERICALCARINE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_POSTCENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PRECENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_PRECUNEUS_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16', ...
                'CTX_LH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'LEFT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16', ...
                'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16', ...
                'LEFT_CAUDATE_UCBERKELEYAV45_10_17_16', ...
                'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16', ...
                'LEFT_PALLIDUM_UCBERKELEYAV45_10_17_16', ...
                'LEFT_PUTAMEN_UCBERKELEYAV45_10_17_16', ...
                'LEFT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16', ...
                'LEFT_VENTRALDC_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_CUNEUS_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_ENTORHINAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_FRONTALPOLE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_FUSIFORM_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_INSULA_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_LINGUAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PARACENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PARSORBITALIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PERICALCARINE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_POSTCENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PRECENTRAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_PRECUNEUS_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16', ...
                'CTX_RH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_AMYGDALA_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_CAUDATE_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_PALLIDUM_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_PUTAMEN_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16', ...
                'RIGHT_VENTRALDC_UCBERKELEYAV45_10_17_16'
                };
            
% select subjects with av45 and cognitive score + av45 ROIs
tadpole_av45_subjects = tadpole(av45_idx, column_names);

%%
AD_idx = strcmpi(tadpole_av45_subjects.DX_bl, 'AD');
EMCI_idx = strcmpi(tadpole_av45_subjects.DX_bl, 'EMCI');
LMCI_idx = strcmpi(tadpole_av45_subjects.DX_bl, 'LMCI');
SMC_idx = strcmpi(tadpole_av45_subjects.DX_bl, 'SMC');
CN_idx = strcmpi(tadpole_av45_subjects.DX_bl, 'CN');

rid = tadpole_av45_subjects.RID;

[unique_rid, ia, ic] = unique(tadpole_av45_subjects.RID);

% count number of visits. unique(i) has rid_counts(i) number of visits.
rid_counts = accumarray(ic,1);
% rid with 3 visits
len3_rid = unique_rid(rid_counts == 3);
% indices of rid with 3 visits
len3_rid_idx = ismember(rid,  len3_rid);
% get subjects with 3 visits
tadpole_av45_len3 = tadpole_av45_subjects(len3_rid_idx,:);


%%
% rids of subjects with 3 visits
rid3 = tadpole_av45_len3.RID;
% number of subjects with 3 visits
num_len3 = length(rid3)/3;

TADPOLE_V3 = {};
[unique_rid, ia, ic] = unique(tadpole_av45_len3.RID);

dx_mat = zeros(num_len3, 3);

for i=1:num_len3
    % index of i'th subject for all 3 visits
    rid3_idx = find(ic == i);
    for j=1:3
        rid3_idx_j = rid3_idx(j);
        TADPOLE_V3{i}{j}.rid = rid3(rid3_idx_j);
        TADPOLE_V3{i}{j}.age = tadpole_av45_len3.AGE(rid3_idx_j);
        TADPOLE_V3{i}{j}.age_bl = tadpole_av45_len3.Years_bl(rid3_idx_j);
        TADPOLE_V3{i}{j}.dx_bl = tadpole_av45_len3.DX_bl(rid3_idx_j);
        TADPOLE_V3{i}{j}.dxchange = tadpole_av45_len3.DXCHANGE(rid3_idx_j);
        TADPOLE_V3{i}{j}.ravlt_immediate = tadpole_av45_len3.RAVLT_immediate(rid3_idx_j);
        TADPOLE_V3{i}{j}.ravlt_learning = tadpole_av45_len3.RAVLT_learning(rid3_idx_j);
        TADPOLE_V3{i}{j}.ravlt_forgetting = tadpole_av45_len3.RAVLT_forgetting(rid3_idx_j);
        TADPOLE_V3{i}{j}.adas13 = tadpole_av45_len3.ADAS13(rid3_idx_j);
        
        % av45.Variables gives the actual values
        TADPOLE_V3{i}{j}.av45 = tadpole_av45_len3(rid3_idx_j, 15:end);
        % 1: NL to NL, 7: MCI to ML, 9: AD to NL
        if TADPOLE_V3{i}{j}.dxchange == 1 || ...
                TADPOLE_V3{i}{j}.dxchange == 7 || ...
                TADPOLE_V3{i}{j}.dxchange == 9 
            dx_current = 1;
        % 2: MCI to MCI, 4: NL to MCI, 8: AD to MCI
        elseif TADPOLE_V3{i}{j}.dxchange == 2 || ...
                TADPOLE_V3{i}{j}.dxchange == 4 || ...
                TADPOLE_V3{i}{j}.dxchange == 8
            dx_current = 2;
        % 3: AD to AD, 5: MCI to AD, 6: CN to AD
        elseif TADPOLE_V3{i}{j}.dxchange == 3 || ...
                TADPOLE_V3{i}{j}.dxchange == 5 || ...
                TADPOLE_V3{i}{j}.dxchange == 6
            dx_current = 3;
        else
            dx_current = -1;
        end
        TADPOLE_V3{i}{j}.dx_current = dx_current;
        
        dx_mat(i,j) = dx_current;
    end
end

% num_111 = sum(sum(dx_mat == [1,1,1], 2) == 3)
% num_112 = sum(sum(dx_mat == [1,1,2], 2) == 3)
% num_113 = sum(sum(dx_mat == [1,1,3], 2) == 3)
% num_122 = sum(sum(dx_mat == [1,2,2], 2) == 3)
% num_123 = sum(sum(dx_mat == [1,2,3], 2) == 3)
% num_223 = sum(sum(dx_mat == [2,2,3], 2) == 3)


% fix the order of age_bl which is +age from baseline
for i=1:num_len3
    tmp_age_bl = zeros(3,1);
    for j=1:3
        tmp_age_bl(j) = TADPOLE_V3{i}{j}.age_bl;
    end
    [~, age_bl_idx] = sort(tmp_age_bl);
    tmp_tadpole = TADPOLE_V3{i};
    for j=1:3
        TADPOLE_V3{i}{j} = tmp_tadpole{age_bl_idx(j)};
    end
end

%%
TADPOLE_V3_dx = {};
counter = 1;
for i=1:num_len3
    if isnan(TADPOLE_V3{i}{1}.adas13) || ...
        isnan(TADPOLE_V3{i}{2}.adas13) || ...
        isnan(TADPOLE_V3{i}{3}.adas13)
        continue
    else
        TADPOLE_V3_dx{counter} = TADPOLE_V3{i};
        counter = counter + 1;
    end
end

%%
N = length(TADPOLE_V3_dx);
T = 3;
% [CN, MCI, AD, age_bl]
Y = zeros(N, T, 1);
X = zeros(N, T, length(TADPOLE_V3{1}{1}.av45.Variables));
for i=1:N
    for t=1:T
        tadpole_i = TADPOLE_V3_dx{i}{t};
        Y(i, t, 1) = tadpole_i.adas13;
%         Y(i, t, 4)  = tadpole_i.age + tadpole_i.age_bl;
        X(i, t, :) = tadpole_i.av45.Variables;
    end
end
av45_names = TADPOLE_V3{1}{1}.av45.Properties.VariableNames;
% check for nan
sum(sum(sum(isnan(X))))
sum(sum(sum(isnan(Y))))

% save('ADAS13_AV45_AV45', 'X');
% save('ADAS13_AV45_ADAS13', 'Y');
% save('AV45_ROI_names', 'av45_names');








