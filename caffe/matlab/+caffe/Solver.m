classdef Solver < handle
  % Wrapper class of caffe::SGDSolver in matlab
  
  properties (Access = private)
    hSolver_self
    attributes
    % attribute fields
    %     hNet_net
    %     hNet_test_nets
  end
  properties (SetAccess = private)
    net
    test_nets
  end
  
  methods
    function self = Solver(varargin)
      % decide whether to construct a solver from solver_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a solver from solver_file
        self = caffe.get_solver(varargin{:});
        return
      end
      % construct a solver from handle
      hSolver_solver = varargin{1};
      CHECK(is_valid_handle(hSolver_solver), 'invalid Solver handle');
      
      % setup self handle and attributes
      self.hSolver_self = hSolver_solver;
      self.attributes = caffe_('solver_get_attr', self.hSolver_self);
      
      % setup net and test_nets
      self.net = caffe.Net(self.attributes.hNet_net);
      self.test_nets = caffe.Net.empty();
      for n = 1:length(self.attributes.hNet_test_nets)
        self.test_nets(n) = caffe.Net(self.attributes.hNet_test_nets(n));
      end
    end
    function h = get_handle( obj )
        h = obj.hSolver_self;
    end
    function iter = iter(self)
      iter = caffe_('solver_get_iter', self.hSolver_self);
    end
    function iter = set_iter(self, iter)
      caffe_('solver_set_iter', self.hSolver_self, iter);
    end
    function max_iter = max_iter(self)
      max_iter = caffe_('solver_get_max_iter', self.hSolver_self);
    end
    function snapshot_prefix = snapshot_prefix(self)
      snapshot_prefix = caffe_('solver_get_snapshot_prefix', self.hSolver_self);
    end
    function restore(self, snapshot_filename)
      CHECK(ischar(snapshot_filename), 'snapshot_filename must be a string');
      CHECK_FILE_EXIST(snapshot_filename);
      caffe_('solver_restore', self.hSolver_self, snapshot_filename);
    end
    function solve(self)
      caffe_('solver_solve', self.hSolver_self);
    end
    function step(self, iters)
      CHECK(isscalar(iters) && iters > 0, 'iters must be positive integer');
      iters = double(iters);
      caffe_('solver_step', self.hSolver_self, iters);
    end
    function reset(self)
      caffe_('reset', self.hSolver_self);
    end
    function update(self)
      caffe_('solver_update', self.hSolver_self);
    end
    function snapshot(self, solverstate_file_noext)
      CHECK(ischar(solverstate_file_noext), 'solverstate_file_noext must be a string');
      [pdir,~,~] = fileparts( solverstate_file_noext );
      if ~exist(pdir,'dir'), mkdir(pdir); end
      caffe_('solver_snapshot', self.hSolver_self, solverstate_file_noext);
    end
    function t = type(self)
      t = caffe_('solver_type', self.hSolver_self);
    end
  end
end