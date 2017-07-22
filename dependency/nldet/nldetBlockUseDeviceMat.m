function nldetBlockUseDeviceMat( block, varargin )

if ismethod( block, 'UseDeviceMat' )
    block.UseDeviceMat( varargin{:} );
end
