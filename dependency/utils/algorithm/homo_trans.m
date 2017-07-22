function T = homo_trans(T)

if size(T,2) == 2
    T = [T,[0;0]];
end

if size(T,1) == 2
    T = [T;0,0,1];
end
