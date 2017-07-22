function s1 = threshold_detection_score( s0, neg_th, pos_th )

pos_idxb = (s0>=pos_th);
if neg_th == pos_th
    neg_idxb = (s0<neg_th);
else
    neg_idxb = (s0<=neg_th);
end

s1 = nan(size(s0),'like',s0);
s1(neg_idxb) = 0;
s1(pos_idxb) = 1;
