function success_list = copy_blobs_from( dst_net, src_net, ...
    renaming_list, is_list_exclusive )
% net = copy_blos_from( dst_net, src_net, renaming_list, is_list_exclusive )
% each row of rename list is {'src_layer_name','target_layer_name (in net)'}

if ~exist('renaming_list','var') || isempty(renaming_list)
    renaming_list = cell(0,2);
end
if size(renaming_list,2)==1
    renaming_list = [renaming_list,renaming_list];
end
if ~exist('is_list_exclusive', 'var') || isempty(is_list_exclusive)
    is_list_exclusive = 0;
end
if strcmp( is_list_exclusive, 'exclusive' )
    is_list_exclusive = 1;
end

if isa(src_net, 'caffe.Net')
    src = src_net;
    Bsrc = src.blob_vec; 
    Nsrc = src.blob_names; Nsrc = reshape(Nsrc,numel(Nsrc),1);
    Bdst = dst_net.blob_vec; 
    Ndst = dst_net.blob_names; Ndst = reshape(Ndst,numel(Ndst),1);
    is_in_src = ismember(renaming_list(:,1),Nsrc);
    is_in_dst = ismember(renaming_list(:,2),Ndst);
    renaming_list(~(is_in_src & is_in_dst),:) = [];
    if is_list_exclusive
        R = renaming_list;
    else
        Lcom = intersect(Nsrc,Ndst);
        is_in_rename = ismember(renaming_list(:,1),Lcom);
        Lcom(is_in_rename) = [];
        R = [Lcom,Lcom;renaming_list];
    end
    Rsrc = R(:,1); Rdst = R(:,2);
    [~,Psrc] = ismember(Rsrc,Nsrc);
    [~,Pdst] = ismember(Rdst,Ndst);
    is_copied = false(length(Psrc),1);
    for k = 1:length(Psrc)
        Bsrc_k = Bsrc(Psrc(k));
        Bdst_k = Bdst(Pdst(k));
        is_same_size = prod(Bsrc_k.shape())==prod(Bsrc_k.shape());
        if ~is_same_size, continue; end

        bd_src=Bsrc_k.get_data();
        bd_dst=Bdst_k.get_data();
        bd_dst(:) = bd_src(:);
        Bdst_k.set_data(bd_dst);
            
        is_copied(k) = true;
    end
    success_list = struct( 'src', Nsrc(Psrc(is_copied)), 'dst', Ndst(Pdst(is_copied)) );
elseif iscell(src_net)
    success_list = cell(size(src_net));
    for k = 1:numel(src_net)
        success_list{k} = caffe.copy_net_from( dst_net, src_net{k}, renaming_list );
    end
else
    error( 'weights_file must be a string or Net or a cell of them' );
end
