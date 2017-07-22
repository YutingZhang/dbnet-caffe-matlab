function T = compact_trans(T)

if size(T,1)==3
    assert( all( T(3,:) == [0,0,1] ), 'not a compact transform' );
    T = T(1:2,:);
end
