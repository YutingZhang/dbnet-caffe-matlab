function varargout = parpool_strong( varargin )

varargout = cell(1,nargout);

p = parpool( varargin{:}, 'IdleTimeout', inf );

[varargout{:}] = edeal(p);

if ~isempty(p)
    f = parfevalOnAll(p,@set_omp_thread_num,0);
    wait(f);
end
