function [theta, delta] = FABRIKc(p_target, z_target, L, initial_theta, initial_delta, kmax, epsilon)

% n = number of segments
n = numel(L); 

% initialise theta and delta
theta = initial_theta(:)'; 
delta = initial_delta(:)';

% initialise geometry
[ptb, pte, ztb, zte, ptj, lt] = initalise_segment(L, theta, delta);

% initial iteration
k = 1;
e = norm(p_target - pte(:, end)); % position error

while e > epsilon && k <= kmax
    % backward pass
    for t = n:-1:1
        if t == n
            pte(:, t) = p_target;
            zte(:, t) = z_target;
        else
            pte(:, t) = ptb(:, t+1);
            zte(:, t) = ztb(:, t+1);
        end
        
        ptj(:, t) = pte(:, t) - lt(t) * zte(:, t);
        if t > 1
            vec = ptj(:, t) - ptj(:, t-1);
            ztb(:, t) = vec / norm(vec);
        else
            ztb(:, t) = [0; 0; 1];
        end
        
        dot_product = dot(ztb(:, t), zte(:, t));
        dot_product = max(min(dot_product, 1), -1);
        theta(t) = acos(dot_product);
        if theta(t) > pi
            error('Not possible: Bending angle exceeds 180 degrees.');
        end
        
        lt(t) = newlinklength(theta(t), L(t));
        ptj(:, t) = pte(:, t) - lt(t) * zte(:, t);
        ptb(:, t) = ptj(:, t) - lt(t) * ztb(:, t);
    end
    
    % forward oass
    for t = 1:n
        if t == 1
            ptb(:, t) = [0; 0; 0];
            ztb(:, t) = [0; 0; 1];
        else
            ptb(:, t) = pte(:, t-1);
            ztb(:, t) = zte(:, t-1);
        end
        
        ptj(:, t) = ptb(:, t) + lt(t) * ztb(:, t);
        if t < n
            vec = ptj(:, t+1) - ptj(:, t);
            zte(:, t) = vec / norm(vec);
        else
            zte(:, t) = z_target;
        end
        
        dot_product = dot(ztb(:, t), zte(:, t));
        dot_product = max(min(dot_product, 1), -1);
        theta(t) = acos(dot_product);
        if theta(t) > pi
            error('Not possible: Bending angle exceeds 180 degrees.');
        end
        
        lt(t) = newlinklength(theta(t), L(t));
        ptj(:, t) = ptb(:, t) + lt(t) * ztb(:, t);
        pte(:, t) = ptj(:, t) + lt(t) * zte(:, t);
    end
    
    e = norm(p_target - pte(:, end));
    k = k + 1;
end

% bending plane angle delta 
for t = 1:n
    rel_pos = pte(:, t) - ptb(:, t);
    delta(t) = atan2(-rel_pos(2), rel_pos(1));
end

if n >= 2
    zb = ztb(:,2);
    zb = zb / norm(zb);
    r  = pte(:,2) - ptb(:,2);
    r_perp = r - dot(r, zb) * zb;
    if norm(r_perp) >= 1e-9
        xref = [1;0;0]; 
        if abs(dot(xref, zb)) > 0.9, xref = [0;1;0]; end
        xb = xref - dot(xref, zb)*zb; xb = xb / norm(xb);
        yb = cross(zb, xb);
        delta(2) = atan2( dot(r_perp, yb), dot(r_perp, xb) );
    end
end



figure;
ax = gca;

set(gcf,'Color','w');                 
set(ax,'Color','w');                  
set(ax,'XColor','k','YColor','k','ZColor','k'); 
grid on; axis equal; view(3);

ax.FontSize = 14;                     
xlabel('X cm','FontSize',16,'Color','k');
ylabel('Y cm','FontSize',16,'Color','k');
zlabel('Z cm','FontSize',16,'Color','k');
ax.GridAlpha = 0.15;


set(gcf,'KeyPressFcn',@(fig,evt) keyPressHandler(evt, ax));

hold on;
for t = 1:n
    plot3(ptb(1,t), ptb(2,t), ptb(3,t), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    plot3(ptj(1,t), ptj(2,t), ptj(3,t), 'gd', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot3(pte(1,t), pte(2,t), pte(3,t), 'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
  
    arc = generateArcUsingPtj(ptb(:,t), ptj(:,t), pte(:,t), L(t), theta(t), 50);
    plot3(arc(1,:), arc(2,:), arc(3,:), 'k-', 'LineWidth', 2);
end

end


function [ptb, pte, ztb, zte, ptj, lt] = initalise_segment(L, theta, delta)
n = numel(L);
ptb = zeros(3, n);
pte = zeros(3, n);
ztb = repmat([0;0;1], 1, n);
zte = zeros(3, n);
ptj = zeros(3, n);
lt = zeros(1, n);

for t = 1:n
    lt(t) = newlinklength(theta(t), L(t));
    if t == 1
        ptb(:, t) = [0; 0; 0];
    else
        ptb(:, t) = pte(:, t-1);
    end
    
    R = rotz(rad2deg(delta(t)))*roty(rad2deg(theta(t)));
    pte_rel = R * [0; 0; L(t)];
    pte(:, t) = ptb(:, t) + pte_rel;
    zte(:, t) = R(:, 3);
    
    ptj(:, t) = ptb(:, t) + lt(t) * ztb(:, t);
end
end

function lt = newlinklength(theta, L)
if theta == 0
    lt = L / 2;
else
    lt = (L/theta) * tan(theta/2);
end
end

function arc = generateArcUsingPtj(ptb, ptj, pte, L, theta, numPoints)
if theta == 0
    arc = [linspace(ptb(1), pte(1), numPoints);
           linspace(ptb(2), pte(2), numPoints);
           linspace(ptb(3), pte(3), numPoints)];
    return;
end

chord = pte - ptb;
d = norm(chord);
midPoint = (ptb + pte) / 2;
e_x = chord / d;

hump_vec = ptj - midPoint;
hump_vec = hump_vec - dot(hump_vec, e_x) * e_x;
if norm(hump_vec) == 0
    if abs(e_x(1)) < abs(e_x(2))
        hump_vec = [1; 0; 0];
    else
        hump_vec = [0; 1; 0];
    end
end
e_y = hump_vec / norm(hump_vec);

R_circle = L / theta;
scale = d / (2 * R_circle * sin(theta/2));

phi = linspace(-theta/2, theta/2, numPoints);
x_local = R_circle * sin(phi) * scale;
y_local = R_circle * cos(phi) - R_circle * cos(theta/2);

arc = zeros(3, numPoints);
for i = 1:numPoints
    arc(:, i) = midPoint + x_local(i)*e_x + y_local(i)*e_y;
end
end

function R = rotz(angle)
angle = deg2rad(angle);
R = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1];
end

function R = roty(angle)
angle = deg2rad(angle);
R = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)];
end


function keyPressHandler(evt, ax)
    switch evt.Key
        case '1'
            view(ax, 0, 90);  
        case '2'
            view(ax, 0, 0);    
        case '3'
            view(ax, 90, 0);  
        case '4'
            view(ax, 3);       
    end
end
