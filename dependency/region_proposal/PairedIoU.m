function S = PairedIoU( boxes1, boxes2 )
% S = PairedIoU( boxes1, boxes2 )
% boxes1 = M * 4, boxes2 = N * 4
% S = M * N;

if ~exist('boxes2','var') || ~size(boxes2,1)
    boxes2 = boxes1;
end

B1 = boxes1(:,1:4).'; 
B2 = boxes2(:,1:4).'; B2 = reshape( B2, [4, 1, size(B2,2)] );

% interEdge = bsxfun( @min, B1([3 4],:,:), B2([3 4],:,:) ) - ...
%     bsxfun( @max, B1([1 2],:,:), B2([1 2],:,:) ) + 1;

interEdge = min( repmat( B1([3 4],:,:), 1, 1, size(B2,3) ), ...
    repmat( B2([3 4],:,:), 1, size(B1,2), 1 ) ) - ...
    max( repmat( B1([1 2],:,:), 1, 1, size(B2,3) ), ...
    repmat( B2([1 2],:,:), 1, size(B1,2), 1 ) ) + 1;
interEdge = max( interEdge, 0 );
interSize = interEdge(1,:,:).*interEdge(2,:,:);


interSize = reshape( interSize, size(interSize,2), size(interSize,3) );

b1Size = ( B1(3,:)-B1(1,:)+1 ) .* ( B1(4,:)-B1(2,:)+1 );
b2Size = ( B2(3,:)-B2(1,:)+1 ) .* ( B2(4,:)-B2(2,:)+1 );

S = interSize./( bsxfun( @plus, b1Size.', b2Size ) - interSize );


% if ~exist('boxes2','var') || ~size(boxes2,1)
% 
%     B = boxes1;
%     S = zeros( size(B,1) );
%     
%     if exist('gcp','file') && ~isempty(gcp('nocreate'))  %matlabpool('size')>0
%         parfor k = 1:size( B,1 )
%             B1=B;
%             sk = zeros(size(B,1),1);
%             sk(k+1:end) = PascalOverlap( B(k,:), B1(k+1:end,:) );
%             S(:,k) = sk;
%         end
%     else
%         for k = 1:size( B,1 )
%             S(k+1:end,k) = PascalOverlap( B(k,:), B(k+1:end,:) );
%         end
%     end
%     S = S + S.' + eye(size(S));
%     
% else
% 
%     flipData = (size(boxes1,1) > size(boxes2,1));
%     if flipData
%         A = boxes2; B = boxes1;
%     else
%         A = boxes1; B = boxes2; 
%     end
% 
%     S = zeros( [ size(B,1), size(A,1)] );
% 
%     if exist('gcp','file') && ~isempty(gcp('nocreate'))  %matlabpool('size')>0
%         parfor k = 1:size( A,1 )
%             S(:,k) = PascalOverlap( A(k,:), B );
%         end
%     else
%         for k = 1:size( A,1 )
%             S(:,k) = PascalOverlap( A(k,:), B );
%         end
%     end
% 
%     if ~flipData
%         S = S.';
%     end
% 
% end
% 
% 
