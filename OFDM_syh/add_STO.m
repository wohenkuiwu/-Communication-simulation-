function y_STO = add_STO(y,nSTO)

if nSTO>=0
    y_STO = [y(nSTO+1:end) zeros(1,nSTO)];
else
    y_STO = [zeros(1,-nSTO) y(1:end+nSTO)];
end