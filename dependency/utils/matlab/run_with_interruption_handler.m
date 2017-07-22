function varargout = run_with_interruption_handler( ...
    func_to_run, func_for_interrupt )

varargout = cell(1,nargout);

interruption_guard = onMutableCleanup( func_for_interrupt );
[varargout{:}] = func_to_run();
interruption_guard.reset();

