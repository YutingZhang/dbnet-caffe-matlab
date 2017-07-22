function s1 = power_detection_score( s0, neg_th, pos_th, reshape_power )

if neg_th == pos_th
    s1 = zeros(size(s0),'like',s0);
    s1(s0<pos_th) = 0;
    return;
end

s1 = min(max((s0-neg_th)./(pos_th-neg_th),0),1);

if reshape_power==1, return; end

s1 = sign(s1).*( abs(s1).^reshape_power );
