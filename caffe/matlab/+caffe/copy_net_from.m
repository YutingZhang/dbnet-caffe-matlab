function success_list = copy_net_from( net, weights_file_or_net, ...
    renaming_list, is_list_exclusive )
% net = copy_net_from( net, weights_file_or_net, renaming_list )
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

if ischar(weights_file_or_net)
    CHECK_FILE_EXIST(weights_file_or_net);
    assert( isempty(renaming_list), 'cannot use renaming for loading from files' );
    caffe_('net_copy_from', net.handle(), weights_file_or_net);
    success_list = [];
elseif isa(weights_file_or_net, 'caffe.Net')
    src = weights_file_or_net;
    Lsrc = src.layer_vec; 
    Nsrc = src.layer_names; Nsrc = reshape(Nsrc,numel(Nsrc),1);
    Ldst = net.layer_vec; 
    Ndst = net.layer_names; Ndst = reshape(Ndst,numel(Ndst),1);
    is_in_src = ismember(renaming_list(:,1),Nsrc);
    is_in_dst = ismember(renaming_list(:,2),Ndst);
    renaming_list(~(is_in_src & is_in_dst),:) = [];
    if is_list_exclusive
        R = renaming_list;
    else
        Lcom = intersect(Nsrc,Ndst);
        is_in_rename = ismember(Lcom,renaming_list(:,1));
        Lcom(is_in_rename) = [];
        R = [Lcom,Lcom;renaming_list];
    end
    Rsrc = R(:,1); Rdst = R(:,2);
    [~,Psrc] = ismember(Rsrc,Nsrc);
    [~,Pdst] = ismember(Rdst,Ndst);
    is_copied = false(length(Psrc),1);
    for k = 1:length(Psrc)
        Bsrc = Lsrc(Psrc(k)).params;
        Bdst = Ldst(Pdst(k)).params;
        if ~numel(Bsrc), continue; end
        if numel(Bsrc)~=numel(Bdst), continue; end
        is_same_size = 1;
        BD_src = cell(numel(Bsrc),1);
        for j = 1:numel(Bsrc)
            BD_src{j} = Bsrc(j).get_data();
            BD_dst_j  = Bdst(j).get_data();
            if numel(BD_src{j})~=numel(BD_dst_j)
                is_same_size = 0;
                break;
            end
        end
        if ~is_same_size, continue; end
        for j = 1:numel(Bsrc)
            Bdst(j).set_data(BD_src{j});
        end
        is_copied(k) = true;
    end
    success_list = struct( 'src', Nsrc(Psrc(is_copied)), 'dst', Ndst(Pdst(is_copied)) );
elseif iscell(weights_file_or_net)
    success_list = cell(size(weights_file_or_net));
    for k = 1:numel(weights_file_or_net)
        success_list{k} = caffe.copy_net_from( net, weights_file_or_net{k}, renaming_list );
    end
else
    error( 'weights_file must be a string or Net or a cell of them' );
end
