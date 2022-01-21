function x=twoD_norm(ori)
ori_std=std(ori(:));
if ori_std~=0
    x=(ori-mean(ori(:)))/ori_std;
else
    x=ori;
end

    