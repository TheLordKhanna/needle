function [ticks_by_motor, status_by_motor] = casecompute1(theta, delta, n, r, l, baseline)

    l1 = (2*n*sin(theta/(2*n))) * ((l/theta) - (r * (sin(delta))));
    l2 = (2*n*sin(theta/(2*n))) * ((l/theta) + (r * sin(pi/3 + delta)));
    l3 = (2*n*sin(theta/(2*n))) * ((l/theta) - (r * cos(pi/6 + delta)));
    lens = [l1, l2, l3];

    if abs(baseline - 6.61) < 1e-9
        motors = [2, 5, 4];   
    elseif abs(baseline - 5.89) < 1e-9
        motors = [6, 3, 1];   
    else
        error('baseline must be exactly 6.61 or 5.89');
    end

    ticks_by_motor  = zeros(1, 6);
    status_by_motor = repmat({''}, 1, 6);

    K = 4096/(pi*4.5);

    for i = 1:3
        motor = motors(i);
        L = lens(i);

        if L < baseline
            status = 'tighten';
        elseif L > baseline
            status = 'loosen';
        else
            status = '';  
        end

        delta_len = abs((L - baseline) * 10);

 
        ticks_unsigned = K * delta_len;


        sgn = 0;
        if ticks_unsigned > 0
            if motor >= 1 && motor <= 3
                if strcmp(status, 'tighten'), sgn = -1; else, sgn = +1; end
            else 
                if strcmp(status, 'tighten'), sgn = +1; else, sgn = -1; end
            end
        end

        ticks_by_motor(motor) = sgn * ticks_unsigned;
        status_by_motor{motor} = status;
    end
end

