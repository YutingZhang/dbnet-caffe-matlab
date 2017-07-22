function a = vec2str( sz_vec )

if isempty(sz_vec)
    a = '';
else
    a = sprintf('%d,',sz_vec);
    a = a(1:(end-1));
end
