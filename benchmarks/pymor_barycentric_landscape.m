function pymor_barycentric_landscape(input_path, output_path, chebfun_path, kernel_path)
% Benchmark official Chebfun kernels on the same data as the Python backends.
% Kernel files are extracted without algorithm changes from a recorded aaa.m.
% Process startup, MAT transfer, extraction and setup are outside timed regions.

addpath(chebfun_path);
addpath(kernel_path);
data = load(input_path);
% Check extracted kernels against public AAA handles, before any timed work.
xcheck = linspace(-1, 1, 51).';
[rcheck, ~, ~, ~, zcheck, fcheck, wcheck] = aaa(exp(xcheck), xcheck, ...
    'mmax', 30, 'lawson', 0, 'cleanup', 0, 'deriv_deg', 1);
test_points = [-0.37; 0.21; zcheck];
assert(max(abs(rcheck{1}(test_points) - pymor_chebfun_reval(test_points, zcheck, fcheck, wcheck))) < 1e-14);
assert(max(abs(rcheck{2}(test_points) - pymor_chebfun_diffbary(test_points, zcheck, fcheck, wcheck, 1))) < 1e-14);
rows = {};
for i = 1:numel(data.cases)
    c = data.cases{i};
    for backend = {'chebfun_aaa', 'chebfun_bary'}
        name = backend{1};
        for operation = {'value', 'derivative'}
            op = operation{1};
            row = struct('case', c.name, 'backend', name, 'operation', op);
            if strcmp(name, 'chebfun_bary') && strcmp(op, 'derivative')
                row.point = struct('status', 'unsupported');
                row.batch = struct('status', 'unsupported');
                row.accuracy = struct();
                row.accuracy_point = struct();
                rows{end + 1} = row;
                continue;
            end
            if strcmp(name, 'chebfun_bary')
                evaluate = @(x) bary(x(:), c.values, c.nodes, c.weights);
            elseif size(c.values, 2) == 1
                % Match AAA's native scalar handles, without a per-entry wrapper.
                if strcmp(op, 'derivative')
                    evaluate = @(x) pymor_chebfun_diffbary(x, c.nodes, c.values, c.weights, 1);
                else
                    evaluate = @(x) pymor_chebfun_reval(x, c.nodes, c.values, c.weights);
                end
            else
                evaluate = @(x) evaluate_entries(x, c.nodes, c.values, c.weights, strcmp(op, 'derivative'));
            end
            if strcmp(op, 'value')
                point_grid = c.point_grid;
                batch_grid = c.value_grid;
            else
                point_grid = c.derivative_grid;
                batch_grid = c.derivative_grid;
            end
            row.point = time_evaluation(evaluate, point_grid, data.repeat, true);
            row.batch = time_evaluation(evaluate, batch_grid, data.repeat, false);
            row.accuracy = struct();
            row.accuracy_point = struct();
            for j = 1:numel(c.accuracy)
                g = c.accuracy{j};
                if strcmp(op, 'value')
                    reference = g.reference_value;
                else
                    reference = g.reference_derivative;
                end
                row.accuracy.(g.name) = accuracy_metrics(evaluate, g.points, reference, false);
                row.accuracy_point.(g.name) = accuracy_metrics(evaluate, g.points, reference, true);
            end
            rows{end + 1} = row;
            fprintf('%s %s %s: point=%s batch=%s\n', c.name, name, op, row.point.status, row.batch.status);
            fflush(stdout);
        end
    end
end
result = struct('octave_version', version(), 'kernel_self_check_passed', true, ...
                'max_recursion_depth', max_recursion_depth(), ...
                'openblas_num_threads', getenv('OPENBLAS_NUM_THREADS'), 'rows', {rows});
try
    result.blas = version('-blas');
catch
    result.blas = 'not reported';
end
fid = fopen(output_path, 'w');
if fid < 0
    error('Cannot open output file: %s', output_path);
end
fprintf(fid, '%s\n', jsonencode(result));
fclose(fid);
end

function values = evaluate_entries(x, nodes, samples, weights, derivative)
values = zeros(numel(x), size(samples, 2));
for j = 1:size(samples, 2)
    if derivative
        values(:, j) = pymor_chebfun_diffbary(x, nodes, samples(:, j), weights, 1);
    else
        values(:, j) = pymor_chebfun_reval(x, nodes, samples(:, j), weights);
    end
end
end

function metrics = accuracy_metrics(evaluate, points, reference, point_mode)
try
    if point_mode
        actual = zeros(size(reference));
        for k = 1:numel(points)
            actual(k, :) = evaluate(points(k));
        end
    else
        actual = reshape(evaluate(points), size(reference));
    end
    bad = sum(~isfinite(actual(:)));
    if bad > 0
        metrics = struct('status', 'nonfinite', 'nonfinite', bad);
    else
        error = abs(actual - reference) ./ max(1, abs(reference));
        % Octave jsonencode rounds some sub-epsilon doubles to zero.
        metrics = struct('status', 'ok', 'nonfinite', 0, ...
                         'max_scaled_error', sprintf('%.17g', max(error(:))), ...
                         'rms_scaled_error', sprintf('%.17g', sqrt(mean(error(:).^2))));
    end
catch err
    metrics = struct('status', 'error', 'message', err.message);
end
end

function result = time_evaluation(evaluate, grid, repeat, point_mode)
try
    evaluate(grid(1));
    durations = zeros(1, repeat);
    for r = 1:repeat
        tic;
        if point_mode
            for j = 1:numel(grid)
                evaluate(grid(j));
            end
        else
            evaluate(grid);
        end
        durations(r) = toc;
    end
    result = struct('status', 'ok', 'median_seconds', median(durations), ...
                    'samples_seconds', durations, 'points', numel(grid), ...
                    'microseconds_per_point', median(durations) / numel(grid) * 1e6);
catch err
    result = struct('status', 'error', 'message', err.message);
end
end
