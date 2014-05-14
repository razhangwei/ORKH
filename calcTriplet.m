function [ triplet ] = calcTriplet( X, label_or_tag, neighborType, K ) 
%generate triplets by combining K nearest same-label neigbhors and K nearest different-labels
%   X: N*D matrix
%   label: 1*D or D*1 label vector
%   K: K nearest neighbors

	switch neighborType	
		case 'label'	
			label = label_or_tag;
		case 'tag'
			tag = label_or_tag;
		otherwise
			error('not supported neighbor types other than label or tag');
	end
	if ~exist('K', 'var')
    K = 10;
  end

  N = size(X, 1);
	step = 1000;
	pos = zeros(N, K);
	neg = zeros(N, K);
	
	tic;
	
	for l = 1 : step : N
		r = min(N, l + step - 1);
		D = calcEuDist( X(l:r, :), X );
		switch neighborType	
			case 'label'	
				idx_neighbor = bsxfun(@eq, label(l:r),  label');
			case 'tag'
				idx_neighbor = tag(l:r, :) * tag' > 0;
			otherwise
				error('not supported neighbor types other than label or tag');
		end
		%same class
	  DD = D;
	  DD(~idx_neighbor) = Inf;
	  DD( logical(eye(size(DD))) ) = Inf;
	  [~, idx] = sort(DD, 2, 'ascend');
	  pos(l : r, 1 : K) = idx(:, 1 : K);
	  
	  %different class
	  DD = D;
	  DD(idx_neighbor) = Inf;
	  [~, idx] = sort(DD, 2, 'descend');
	  neg(l:r, 1:K) = idx(:, 1:K);	  

	  if mod(r, 10000) == 0
	  	fprintf('%d...', r);
	  	toc; tic;
	  end		
	end

	triplet = zeros(N*K*K, 3);	
	for i = 1 : N		
		ID1 = i * ones(K, K);
		[ID2, ID3] = ndgrid(pos(i, :), neg(i, :));    
		triplet( (i-1)*K^2 + (1 : K^2) , : ) = [ID1(:), ID2(:), ID3(:)];
	end
	triplet = triplet(randperm(size(triplet, 1)), :);
end

