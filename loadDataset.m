% load dataset data into memory, and perform some data preprocessing
function dataset = loadDataset (task, dataset)

  % load dataset
  fprintf('Loading dataset %s\n', dataset.name);
  temp = load([dataset.path dataset.filename]);
  dataset = copyField(dataset, temp);

  % rename label variable
  if isfield(dataset, 'labelName')
    dataset = renameField(dataset, dataset.labelName, 'label');
  end

  % rename value variable
  if isfield(dataset, 'valueName')
    dataset = renameField(dataset, dataset.valueName, 'value');
  end

  % subsample
  if ~isfield(dataset, 'subsample')
    dataset.subsample = size(dataset.X, 1);
  end
  dataset = subsample(task, dataset);

  % partition into training, testing, and validation set
  fprintf('Partitioning dataset %s into training, testing, and validation set\n', dataset.name);
  if ~isfield(dataset, 'numValidation')
    dataset.numValidation = dataset.numTest;
  end
  N = size(dataset.X, 1);
  N2 = dataset.numTest;
  N3 = dataset.numValidation;
  N1 = N - N2 - N3;
  dataset.indexTrain = [1: N1]';
  dataset.indexTest = [N1 + 1: N1 + N2]';
  dataset.indexValidation = [N1 + N2 + 1: N]';
  fprintf('  # of training data: %d\n', N1);
  fprintf('  # of testing data: %d\n', N2);
  fprintf('  # of validation data: %d\n', N3);

  % feature scaling
  if isfield(dataset, 'normFilter')
    fprintf('Feature scaling on dataset %s\n', dataset.name);
    dataset = featureScaling(dataset);
  end
  
  % compute neighbor threshold
  if ismember(dataset.neighborType, {'dist', 'affinity', 'value'})
    fprintf('Computing neighbor threshold for dataset %s with target of average %d neighbors for each query\n', dataset.name, dataset.aveNeighbor);
    cacheFile = sprintf('%s/neighborThresh_%s.mat', task.dataDir, dataset.name);
    if loadCache(cacheFile, task.forceFresh, getConst('CACHE_VER_NEIGHBOR_THRESH'), @updaterNeighborThresh)
      tp = timerStart();
      threshold = calcNeighborThresh(dataset);
      timeCost = timerStop(tp);
      save(cacheFile, 'version', 'threshold', 'timeCost', '-v7.3');
    end
    dataset.neighborThreshold = threshold;
    fprintf('  Threshold: %.6g\n', threshold);
    fprintf('  Time cost (elapsed): %.4gs\n', timeCost.etime);
    if (isfield(timeCost, 'ctime'))
      fprintf('  Time cost (CPU): %.4gs\n', timeCost.ctime);
    end
  end

  % computing ground-truth neighbor
  fprintf('Computing ground-truth neighbor for dataset %s\n', dataset.name);
  cacheFile = sprintf('%s/neighbor_%s.mat', task.dataDir, dataset.name);
  if loadCache(cacheFile, task.forceFresh, getConst('CACHE_VER_NEIGHBOR'), @updaterNeighbor)
    tp = timerStart();
    neighborTest = calcNeighbor(dataset, dataset.indexTest, dataset.indexTrain);
    neighborValidation = calcNeighbor(dataset, dataset.indexValidation, dataset.indexTrain);
    timeCost = timerStop(tp);
    save(cacheFile, 'version', 'neighborTest', 'neighborValidation', 'timeCost', '-v7.3');
  end
  dataset.neighborTest = neighborTest;
  dataset.neighborValidation = neighborValidation;
  aveNeighborTest = mean(sum(neighborTest, 2));
  aveNeighborValidation = mean(sum(neighborValidation, 2));
  fprintf('  Average neighbor (testing): %.2f (%.2f%%)\n', aveNeighborTest, aveNeighborTest / N1 * 100);
  fprintf('  Average neighbor (validation): %.2f (%.2f%%)\n', aveNeighborValidation, aveNeighborValidation / N1 * 100);
  fprintf('  Time cost (elapsed): %.4gs\n', timeCost.etime);
  if (isfield(timeCost, 'ctime'))
    fprintf('  Time cost (CPU): %.4gs\n', timeCost.ctime);
  end

  % sample training triplet
  fprintf('Generating triplets for dataset %s\n', dataset.name);
  cacheFile = sprintf('%s/triplet_%s.mat', task.dataDir, dataset.name);
  if loadCache(cacheFile, task.forceFresh, getConst('CACHE_VER_TRIPLET') )
    tp = timerStart();
    switch dataset.neighborType
      case 'label'
        triplet = calcTriplet(dataset.X(dataset.indexTrain, :), dataset.label(dataset.indexTrain), 'label') ;
      case 'tag'
        triplet = calcTriplet(dataset.X(dataset.indexTrain, :), dataset.tag(dataset.indexTrain), 'tag');  
      otherwise
        error('Not supported to generating triplets for dataset %s yet\n', dataset.name);
    end    
    timeCost = timerStop(tp);
    save(cacheFile, 'version', 'triplet', 'timeCost', '-v7.3');
  end
  dataset.triplet = triplet;  
  fprintf(' # of tuplets: %d\n', size(dataset.triplet, 1));
  fprintf(' Average # of same-class and diff-class neighbors: %d\n', round(sqrt(size(dataset.triplet,1)/ length(dataset.indexTrain))));
  if (isfield(timeCost, 'ctime'))
    fprintf('  Time cost (CPU): %.4gs\n', timeCost.ctime);
  end

end
