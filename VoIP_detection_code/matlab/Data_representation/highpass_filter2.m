function out1=highpass_filter2(y,need_norm,need_hp)
if need_hp==1
    y1=y;
    len = length(y1);
    y_addzero = [0;y1;0];
    ylen = len+2;
    out = y_addzero(2:ylen-1) - 0.5*y_addzero(1:ylen-2) - 0.5*y_addzero(3:ylen);
else
    out=y;
end
x=out;
if need_norm==1
    standardVar = std(x(:));
    if (standardVar~=0) 
        out1 = (x-mean(x(:)))/standardVar;
    else
        out1=x;
    end;
else
    out1=x;
end
