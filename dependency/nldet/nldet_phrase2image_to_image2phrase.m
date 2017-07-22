function image2phrases = nldet_phrase2image_to_image2phrase( phrases2images, IMG_IND )

if isvector( phrases2images )
    image2phrases = phrases2images;
    return;
end

I = codebook_cols( IMG_IND, {'image_id','image_order'} );

t1 = tic_print( 'Build image2phrases index : \n' );
n  = size(phrases2images,2);

image2phrases = cell( 1,size(I,1) );

for k = 1:n

    tic_toc_print( '  - image2phrases: %d / %d\n', k, n );

    phrase_id  = phrases2images{1,k};
    if ischar(phrase_id)
        phrase_id = str2double(phrase_id);
    end
    image_info = phrases2images{2,k};
    image_info.pos = vec(image_info.pos).';
    image_info.neg = vec(image_info.neg).';

    image_ids  = [image_info.pos,image_info.neg];
    image_orders = map_with_codebook( image_ids, I );

    for j = 1:numel(image_orders)
        image2phrases{image_orders(j)} = [ ...
            image2phrases{image_orders(j)}, phrase_id ];
    end

end
toc_print(t1);
