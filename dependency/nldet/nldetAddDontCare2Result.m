function R1 = nldetAddDontCare2Result(R, text_comp_func, sim_th)

if nargin<2
    R1 = R;
    for k = 1:numel(R)
        R1(k).dontcare = false( size(R(k).is_gt) );
    end
    return;
end

R1 = R;
for k = 1:numel(R)
    text_ids = [R(k).text.source];
    tsim  = text_comp_func( text_ids, text_ids );
    tcomp = (tsim>=sim_th);
    R1(k).dontcare = nldet_donotcare_from_gt_and_textsim( tcomp, R(k).is_gt );
end
