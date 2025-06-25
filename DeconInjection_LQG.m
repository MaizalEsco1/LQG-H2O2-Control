%% LQG UNTUK SISTEM INJEKSI H?O? DENGAN TRACKING DINAMIS & ROBUSTNESS
clc; clear; close all;

%% STEP 1: BACA DATA CSV
filename = 'C:\Users\Thinkpad X280\OneDrive\ITS\Lecture Note\Semester 2\3 Advance Automatic Control\EAS\Data Decon model 3Var.csv';
raw = readtable(filename, 'ReadVariableNames', false);
data = raw(2:end, :);  % abaikan header

t = str2double(data{:,1});     % waktu [menit]
u = str2double(data{:,2});     % input: pump speed [ml/min]
y = str2double(data{:,3});     % output: H?O? level [ppm]

%% STEP 2: PLOT DATA INPUT DAN OUTPUT
figure;
yyaxis left
plot(t, u, 'b--', 'LineWidth', 1.5);
ylabel('Pump Speed [ml/min]');
ylim([0 max(u)*1.1]);

yyaxis right
plot(t, y, 'r-', 'LineWidth', 1.5);
ylabel('H?O? Concentration [ppm]');
ylim([0 max(y)*1.1]);

xlabel('Time [minutes]');
title('Data Sistem Injeksi H?O?');
legend('Pump Speed', 'H?O? Level', 'Location', 'northwest');
grid on;

%% STEP 3: PERSIAPAN DATA IDENTIFIKASI
valid = ~isnan(u) & ~isnan(y);
u_clean = u(valid);
y_clean = y(valid);
Ts = 1;

n_pad = 5;
u_padded = [zeros(n_pad,1); u_clean];
y_padded = [repmat(y_clean(1), n_pad, 1); y_clean];
data_id = iddata(y_padded, u_padded, Ts);

%% STEP 4: IDENTIFIKASI SISTEM (N4SID)
order = 2;
sys = n4sid(data_id, order);
[A, B, C, D] = ssdata(sys);

disp('=== Matriks State-Space ===');
disp('A ='); disp(A);
disp('B ='); disp(B);
disp('C ='); disp(C);
disp('D ='); disp(D);

%% STEP 5: VALIDASI MODEL
y_model = sim(sys, data_id.InputData);
y_actual = data_id.OutputData;
fit_percent = 100 * (1 - norm(y_actual - y_model) / norm(y_actual - mean(y_actual)));

t_valid = (0:length(y_model)-1) * Ts;
figure;
plot(t_valid, y_actual, 'k', 'LineWidth', 1.5); hold on;
plot(t_valid, y_model, 'r--', 'LineWidth', 1.5);
xlabel('Time [minutes]');
ylabel('H?O? Level [ppm]');
title('Validasi Model Identifikasi');
legend({'Output Aktual', ['Output Model (Fit = ', num2str(round(fit_percent,2)), '%)']});
grid on;

%% STEP 6: DESAIN LQR
% Tuning variabel penalti kontrol dan state
q1 = 10;     % penalti untuk x1 (H2O2)
q2 = 1;      % penalti untuk x2 (pump speed)
r_val = 1;   % penalti untuk u (kontrol)

Q = diag([q1, q2]);
R = r_val;

[K, ~, eig_lqr] = lqr(A, B, Q, R);
if size(K,2) == 1
    K = K';
end

% Feedforward gain untuk tracking referensi konstan
r = 500;  % setpoint tetap
N = -inv(C * inv(A - B*K) * B);

%% STEP 7: DESAIN KALMAN FILTER (tanpa gangguan)
W = 0.01 * eye(2);  % noise proses
V = 0.1;            % noise pengukuran

[L, ~, eig_obs] = lqe(A, eye(2), C, W, V);

%% STEP 8: SIMULASI SISTEM LQG
A_cl = [A - B*K, B*K;
        zeros(size(A)), A - L*C];

B_cl = [B*N;
         zeros(size(B))];

x0 = [1; 0];         % initial actual state
xhat0 = [0; 0];      % initial estimate
x_aug0 = [x0; xhat0];

tspan = 0:0.1:30;

[~, x_aug] = ode45(@(t,x) A_cl * x + B_cl * r, tspan, x_aug0);
x_aug = real(x_aug);

x_true = x_aug(:,1:2);
x_est  = x_aug(:,3:4);

y_out = x_true * C';                     % output H2O2
u_lqg = -x_est * K' + N * r;             % kontrol dari estimasi

%% STEP 9: PLOTTING OUTPUT DAN SINYAL KONTROL
figure;


subplot(2,1,1);
plot(tspan, y_out, 'b', 'LineWidth', 1.5); hold on;
yline(r, 'k--', 'LineWidth', 1.2);  % Garis setpoint
ylabel('H2O2 Level [ppm]');
title(['Output Response (Q = diag([', num2str(q1), ',', num2str(q2), ']), R = ', num2str(r_val), ')']);
legend('Output y(t)', ['Setpoint = ' num2str(r) ' ppm'], 'Location', 'best');
grid on;

subplot(2,1,2);
plot(tspan, u_lqg, 'r', 'LineWidth', 1.5);
xlabel('Time [minutes]');
ylabel('Control Input u(t) [ml/min]');
title('Control Signal from LQG (Kalman + LQR)');
grid on;




%% STEP 10: INFO TAMBAHAN
disp('=== Gain LQR K ==='); disp(K);
disp('=== Gain Kalman L ==='); disp(L);
disp('=== Feedforward Gain N ==='); disp(N);
disp(['Fit Identifikasi Model = ', num2str(round(fit_percent,2)), ' %']);
disp('=== Eigenvalue sistem tertutup (LQR) ==='); disp(eig_lqr);
disp('=== Eigenvalue observer (Kalman) ==='); disp(eig_obs);

%% FUNGSI DYNAMIK AUGMENTED UNTUK ODE
function dxdt = LQG_augmented_dynamics(t, x, A_cl, B_cl, r_func, disturbance)
    r = r_func(t);
    w = disturbance(t);  % gangguan acak
    dxdt = A_cl * x + B_cl * r;
    dxdt(1:2) = dxdt(1:2) + w;  % tambahkan gangguan hanya pada x (state aktual)
end
