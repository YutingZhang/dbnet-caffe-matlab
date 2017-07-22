% this class is for the ENUM in prototxt
classdef proto_enum_class
    properties ( Access = private )
        enum_val
    end
    methods
        function obj = proto_enum_class( enum_val )
            obj.enum_val = enum_val;
        end
        function disp(obj)
            % must use disp(sprintf(...)) rather than fprintf to display it
            % correctly in format
            if ischar(obj.enum_val)
                disp(sprintf('[enum] %s\n', obj.enum_val));
            else
                disp(sprintf('[enum:%s] %s\n', ...
                    class(obj.enum_val), any2str(obj.enum_val)));
            end
        end
        function v = val(obj)
            v = obj.enum_val;
        end
        function e = eq(a,b)
            if isa( a, 'proto_enum_class' ), a = a.val(); end
            if isa( b, 'proto_enum_class' ), b = b.val(); end
            e = strcmp(a,b);
        end
    end
end
