function t = nldetGetRegionProposalInputType( SpecificDir4RegionProposalNet )

if ~exist('SpecificDir4RegionProposalNet','var')
    SpecificDir4RegionProposalNet = evalin( 'caller', 'SPECIFIC_DIRS.RegionProposalNet' );
end

S = prototxt2struct( fullfile(SpecificDir4RegionProposalNet,'type.prototxt'), [], 0 );
t = S.input_type;
