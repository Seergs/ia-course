clear all
close all
clc

% Funcion Objetivo
function error_evaluation = f(p,pf)
     error_evaluation = norm(pf-p);
end

%Funcion Objetivo con Penalizacion
function fitness = fd(p,pf,x,xl,xu)
    fitness = f(p,pf) + (1000 * Penalty(x,xl,xu));
end

%Funcion para Penalizacion
function z = Penalty (x,xl,xu)
   z = 0;
   m = numel(xl);
   
   for i=1:m
       if xl(i) < x(i)
           z = z + 0;
       else
           z = z + (x(i)-xl(i))^2;
       end
       
       if x(i) < xu(i)
           z = z + 0;
       else
           z = z + (x(i)-xu(i))^2;
       end
   end
end

%Modelo Cinematico Directo
function p = FK(q,l1,l2)
    p = [0.0; 0.0; 0.0];
    p(1) = -sin(q(1)-pi/2)*(l1*cos(q(2))+l2*cos(q(2)+q(3)));
    p(2) = cos(q(1)-pi/2)*(l1*cos(q(2))+l2*cos(q(2)+q(3)));
    p(3) = l1*sin(q(2))+l2*sin(q(2)+q(3));
end

%Funcion para Dibujar Manipulador
function Dibujar_Manipulador (q,l1,l2,pf)
    % Dibujar manipulador
    T = @(theta,a,d,alpha) ...
         [cos(theta) -sin(theta)*cos(alpha) sin(theta)*sin(alpha) a*cos(theta);...
          sin(theta) cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);...
          0 sin(alpha) cos(alpha) d; 0 0 0 1];
    
    T1 = T(q(1),0.0,0.0,pi/2);
    T2 = T(q(2),l1,0.0,0.0);
    T3 = T(q(3),l2,0.0,0.0);
    
    T12 = T1*T2;
    T13 = T12*T3;
    
    p1 = T1(1:3,4);
    p2 = T12(1:3,4);
    p3 = T13(1:3,4);
    
    figure
    grid on
    hold on
    view([-35,35])
    
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([-1.2 1.2 -1.2 1.2 -1 1]);
    
    plot3(p1(1),p1(2),p1(3),'bo','MarkerSize',15,'LineWidth',4)
    plot3(p2(1),p2(2),p2(3),'bo','MarkerSize',15,'LineWidth',4)
    plot3(p3(1),p3(2),p3(3),'bo','MarkerSize',15,'LineWidth',4)

    line([p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'color',[0 0 1],'LineWidth',5)
    line([p2(1) p3(1)],[p2(2) p3(2)],[p2(3) p3(3)],'color',[0 0 1],'LineWidth',5)
    
    plot3(pf(1),pf(2),pf(3),'rx','MarkerSize',15,'LineWidth',4) 
end

% ------------------ Definición de variables necesarias-------------------

% La posición final que se espera que el programa
% encuentre mediante un algoritmo inteligente
p_final = [0.5; 0.1; 0.3];

% Límites del espacio de búsqueda
xl = [-160; -150; -135] * (pi/180);
xu = [160; 150; 135] * (pi/180);

% Medidas del eslabón del manipulador
l1 = 0.5;
l2 = 0.5;

% Número de iteraciones
generations = 200;
% Tamaño de la población
population_size = 50;
% Dimensión del problema
dimension = 3;

% Factor de amplificación
F = 0.6;
% Constante de recombinación
CR = 0.9; 


% Esta matriz contiene a las posibles soluciones del problema
x = zeros(dimension,population_size);
% Este vector contiene las aptitutes de las soluciones
fitness = zeros(population_size,1);


%--------------------- Programa principal --------------------------


%Inicializacion de la población con individuos aleatorios
for i=1:population_size
    x(:,i) = xl+(xu-xl).*rand(dimension,1);
    
    q = x(:,i);
    
    % Se calcula un vector mediante cinemática directa
    p = FK(q,l1,l2);
    
    % Se calcula la aptitud del vector resultante de la cinemática directa.
    % Aquí se usa la función objetivo con penalización para lograr obtener
    % soluciones dentro del espacio de búsqueda.
    fitness(i) = fd(p,p_final,q,xl,xu);
end


% Iterando por el número de generaciones
for n=1:generations
    disp(n);
    for i=1:population_size
        % Se obtienen tres individuos aleatorios de la población tales que
        % r1 =! r2 != r3 != i
        r1  = i;
        while r1==i
            r1 = randi([1,population_size]);
        end

        r2  = r1;
        while r2==r1 || r2==i
            r2 = randi([1,population_size]);
        end

        r3  = r2;
        while r3==r2 || r3==r1 || r3==i
            r3 = randi([1,population_size]);
        end

        % Creación del vector mutante
        v = x(:,r1) + (F*(x(:,r2)-x(:,r3)));

        
        % Proceso de recombinación
        u = zeros(dimension,1);
        for j=1:dimension
            r = rand;

            if r<= CR
                u(j) = v(j);
            else
                u(j) = x(j,i);
            end
        end

        % Proceso de selección, se hace en base a la aptitud del individuo
        q = u;
        p = FK(q,l1,l2);
        if fd(p,p_final,q,xl,xu) < fitness(i)
            x(:,i) = q;
            fitness(i) = fd(p,p_final,q,xl,xu);
        end
    end
end

% Indice del individuo con mejor aptitud
[~,ig] = min(fitness);
q = x(:,ig);

%Impresion de Resultados
disp('----Mínimo global----');
disp(q);
disp('----Vector de Posiciones (Grados)----');
disp(q*(180/pi));
disp('-----Posicion final del Actuador-----');
disp(FK(q,l1,l2));

% Se grafica y muestra el manipulador en su posición final.
Dibujar_Manipulador(q,l1,l2,p_final);




