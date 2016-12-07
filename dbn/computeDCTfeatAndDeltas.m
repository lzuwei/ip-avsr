function dctFeatures = computeDCTfeatAndDeltas(dataMatrix, w, h, noCoeff)
    dctFeatures = computeDCTfeat(dataMatrix, w, h, noCoeff);
    d1 = deltas(dctFeatures, 9);
    d2 = deltas(d1, 9);
    dctFeatures = horzcat(dctFeatures, d1, d2);
end
