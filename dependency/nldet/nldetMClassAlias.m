function a = nldetMClassAlias(a)

a = map_with_codebook( a, nldetMClassCodebook );

function A = nldetMClassCodebook()

A = { ...
    '$rpn_cached', 'nldetRegionProposalNet_Cached'
    '$rpn_empty', 'nldetRegionProposalNet_Empty'
    '$rfn', 'nldetRegionFeatNet' 
    '$tfn', 'nldetTextFeatNet'
    '$prn_glogistic', 'nldetPairNet_GeneralLogistic'
    };
