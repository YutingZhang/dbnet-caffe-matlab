classdef nldetCaffeBlock < handle

    properties (GetAccess=public, SetAccess=protected)
        block
    end
    
    methods
       
        function obj = nldetCaffeBlock( block )
            obj.block = block;
        end
        
        function SetInputData( obj, varargin )
            obj.SetInputData_( varargin{:} );
        end
        
        function varargout = GetOutputData( obj )
            varargout = cell(1,nargout);
            [varargout{:}] = obj.GetOutputData_();
        end
        
        function SetOutputDiff( obj, varargin )
            obj.SetOutputDiff_( varargin{:} );
        end

        function varargout = GetInputDiff( obj )
            varargout = cell(1,nargout);
            [varargout{:}] = obj.GetInputDiff_();
        end
        
        function Forward( obj )
            obj.Forward_();
        end
        
        function Backward( obj )
            obj.Backward_();
        end

        function ForwardBackward( obj )
            obj.ForwardBackward_();
        end
        
        function Update( obj )
            obj.Update_();
        end
        
        function Save( obj, varargin )
            obj.Save_( varargin{:} );
        end

        function Load( obj, varargin )
            obj.Load_( varargin{:} );
        end
        
        function TryLoad( obj, varargin )
            obj.TryLoad_( varargin{:} );
        end
        
        function TryLoad_Solver( obj, varargin )
            obj.TryLoad_Solver_( varargin{:} );
        end
        
        function UseDeviceMat( obj, varargin )
            obj.UseDeviceMat_( varargin{:} );
        end
        
        function UseCPUMat( obj )
            obj.UseCPUMat_();
        end
        
        function ToggleInputDiff( obj, varargin )
            obj.ToggleInputDiff_( varargin{:} );
        end
        
        function SetMemoryEcon( obj, varargin )
            obj.SetMemoryEcon_( varargin{:} );
        end
        
    end
    
    methods (Access=protected)
        function SetInputData_( obj, varargin )
            obj.block.SetInputData( varargin{:} );
        end
        
        function varargout = GetOutputData_( obj )
            varargout = cell(1,nargout);
            [varargout{:}] = obj.block.GetOutputData();
        end
        
        function SetOutputDiff_( obj, varargin )
            obj.block.SetOutputDiff( varargin{:} );
        end

        function varargout = GetInputDiff_( obj )
            varargout = cell(1,nargout);
            [varargout{:}] = obj.block.GetInputDiff();
        end
        
        function Forward_( obj )
            obj.block.Forward();
        end
        
        function Backward_( obj )
            obj.block.Backward();
        end

        function ForwardBackward_( obj )
            obj.block.ForwardBackward();
        end
        
        function Update_( obj )
            obj.block.Update();
        end
    
        function Save_( obj, varargin )
            obj.block.Save( varargin{:} );
        end

        function Load_( obj, varargin )
            obj.block.Load( varargin{:} );
        end
        
        function TryLoad_( obj, varargin )
            obj.block.TryLoad( varargin{:} );
        end
        
        function TryLoad_Solver_( obj, varargin )
            obj.block.TryLoad_Solver( varargin{:} );
        end
        
        function UseDeviceMat_( obj, varargin )
            obj.block.UseDeviceMat( varargin{:} );
        end
        
        function UseCPUMat_( obj )
            obj.block.UseCPUMat();
        end
        
        function ToggleInputDiff_( obj, varargin )
            obj.block.ToggleInputDiff( varargin{:} );
        end
        
        function SetMemoryEcon_( obj, varargin )
            obj.block.SetMemoryEcon( varargin{:} );
        end

    end
    
end
