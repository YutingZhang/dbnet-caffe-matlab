function mkpdir_p( p )

PDIR = fileparts(p);
if ~isempty(PDIR)
    mkdir_p( PDIR );
end

end
