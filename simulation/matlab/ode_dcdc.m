
function dxdt = ode_dcdc(t,s,u)
    % parameter initialization
    xc=70;
    xl=3;
    rc=0.005;
    rl=0.05;
    ro=1;
    vs=1;

    dxdt = zeros(2,1);
    switch u
        case 1
            dxdt(1)=-rl/xl*s(1)+vs/xl;
			dxdt(2)=-1/(xc*(ro+rc))*s(2);
        case 2
            dxdt(1)=-(1/xl)*(rl+ro*rc/(ro+rc))*s(1)-(1/xl)*ro/(5*(ro+rc))*s(2)+vs/xl;
            dxdt(2)=(1/xc)*5*ro/(ro+rc)*s(1)-(1/xc)*(1/(ro+rc))*s(2);
    end

end
