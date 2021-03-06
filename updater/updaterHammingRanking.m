function cache = updaterHammingRanking (cache, verCur, verTarget)

  if verCur < 1
    temp = cache.timeCost;
    cache.timeCost = struct;
    cache.timeCost.etime = temp;
  end

  if verCur < 2
    if isfield(cache.timeCost, 'ctime')
      timeCostNew.etime = cache.timeCost.ctime;
      timeCostNew.ctime = cache.timeCost.etime;
      cache.timeCost = timeCostNew;
    end
  end

  cache.version = verTarget;

end
