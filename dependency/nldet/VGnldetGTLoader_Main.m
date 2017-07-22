classdef VGnldetGTLoader_Main < VGnldetGTLoader
    properties ( Access = protected )
    end
    methods
        function obj = VGnldetGTLoader_Main_Plus_ObjRelObj( DataSubset_Name, ...
                SpecificDir4PrepDataset )
            PARAM = struct();
            PARAM.ann_name_regions = {'region_descriptions'};
            obj@VGnldetGTLoader( DataSubset_Name, ...
                SpecificDir4PrepDataset, PARAM );
        end 
    end
end

