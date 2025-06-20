$(document).ready(function() {
    let stockData = null;

    // Initialize date range picker
    var end = moment();
    var start = moment().subtract(5, 'years');
    
    $('#dateRange').daterangepicker({
        startDate: start,
        endDate: end,
        ranges: {
           'Last 30 Days': [moment().subtract(29, 'days'), moment()],
           'Last 3 Months': [moment().subtract(3, 'months'), moment()],
           'Last 6 Months': [moment().subtract(6, 'months'), moment()],
           'Last 1 Year': [moment().subtract(1, 'year'), moment()],
           'Last 5 Years': [moment().subtract(5, 'years'), moment()]
        }
    });
    
    // Fetch data button click handler
    $('#fetchDataBtn').click(function() {
        const ticker = $('#tickerInput').val().trim().toUpperCase();
        if (!ticker) {
            alert('Please enter a valid ticker symbol');
            return;
        }
        
        const dateRange = $('#dateRange').val().split(' - ');
        const startDate = moment(dateRange[0], 'MM/DD/YYYY').format('YYYY-MM-DD');
        const endDate = moment(dateRange[1], 'MM/DD/YYYY').format('YYYY-MM-DD');
        
        // Show loading spinner - UPDATED
        $('.loading').addClass('show');
        $('#loadingText').html(`<span class="loading-dots">Fetching data for ${ticker}</span>`);
        
        // Fetch stock data
        $.ajax({
            url: '/fetch_stock_data',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                ticker: ticker,
                start_date: startDate,
                end_date: endDate
            }),
            success: function(response) {
                if (response.status === 'success') {
                    // Store data
                    stockData = response;
                    
                    // Generate plots
                    generatePlots(ticker, response);
                    
                    // Show data sections
                    $('#dataOverviewSection').show();
                    $('#timeSeriesAnalysisSection').show();
                    $('#forecastingSection').show();
                    
                    // Hide forecast results section
                    $('#forecastResults').hide();
                    
                    // Update statistics
                    updateStatistics(ticker, response);
                } else {
                    alert(`Error: ${response.message}`);
                }
                
                // Hide loading spinner - UPDATED
                $('.loading').removeClass('show');
            },
            error: function(xhr, status, error) {
                $('.loading').removeClass('show'); // UPDATED
                alert('Error fetching stock data: ' + error);
            }
        });
    });
    
    // Generate plots function
    function generatePlots(ticker, data) {
        // Show loading spinner - UPDATED
        $('.loading').addClass('show');
        $('#loadingText').html(`<span class="loading-dots">Generating visualizations</span>`);
        
        $.ajax({
            url: '/generate_plots',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                ticker: ticker,
                dates: data.dates,
                prices: data.prices,
                volume: data.volume,
                returns: data.returns,
                ma_50: data.ma_50,
                ma_200: data.ma_200
            }),
            success: function(response) {
                if (response.status === 'success') {
                    // Update plot images
                    $('#pricePlot').attr('src', 'data:image/png;base64,' + response.plots.price_plot);
                    $('#returnsPlot').attr('src', 'data:image/png;base64,' + response.plots.returns_plot);
                    $('#volumePlot').attr('src', 'data:image/png;base64,' + response.plots.volume_plot);
                } else {
                    alert(`Error: ${response.message}`);
                }
                
                // Hide loading spinner - UPDATED
                $('.loading').removeClass('show');
            },
            error: function(xhr, status, error) {
                $('.loading').removeClass('show'); // UPDATED
                alert('Error generating plots: ' + error);
            }
        });
    }
    
    // Fix statistics function with better data handling
    function updateStatistics(ticker, data) {
        console.log('Updating statistics with data:', data);
        
        try {
            // Handle nested prices data structure
            let prices = [];
            const rawPrices = data.prices;
            
            for (let i = 0; i < rawPrices.length; i++) {
                let price_item = rawPrices[i];
                if (Array.isArray(price_item)) {
                    if (price_item.length > 0) {
                        prices.push(parseFloat(price_item[0]));
                    }
                } else if (price_item !== null && price_item !== undefined) {
                    prices.push(parseFloat(price_item));
                }
            }
            
            prices = prices.filter(p => !isNaN(p) && p > 0);
            
            if (prices.length === 0) {
                console.error('No valid prices found');
                return;
            }
            
            // Handle returns data
            let returns = [];
            if (data.returns && Array.isArray(data.returns)) {
                for (let i = 0; i < data.returns.length; i++) {
                    let return_item = data.returns[i];
                    if (Array.isArray(return_item)) {
                        if (return_item.length > 0) {
                            returns.push(parseFloat(return_item[0]));
                        }
                    } else if (return_item !== null && return_item !== undefined) {
                        returns.push(parseFloat(return_item));
                    }
                }
            }
            
            returns = returns.filter(r => !isNaN(r));
            
            // Calculate statistics
            const currentPrice = prices[prices.length - 1];
            const previousPrice = prices[prices.length - 2];
            const priceChange = previousPrice ? ((currentPrice - previousPrice) / previousPrice * 100) : 0;
            const yearHigh = Math.max(...prices);
            const yearLow = Math.min(...prices);
            
            // Update metric cards
            $('#currentPrice').text(`$${currentPrice.toFixed(2)}`);
            
            const changeElement = $('#priceChange');
            if (priceChange >= 0) {
                changeElement.text(`+${priceChange.toFixed(2)}%`)
                    .removeClass('text-danger')
                    .addClass('text-white');
            } else {
                changeElement.text(`${priceChange.toFixed(2)}%`)
                    .removeClass('text-success')
                    .addClass('text-white');
            }
            
            $('#yearHigh').text(`$${yearHigh.toFixed(2)}`);
            $('#yearLow').text(`$${yearLow.toFixed(2)}`);
            
            // Calculate additional statistics
            const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
            const variance = prices.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / prices.length;
            const stdDev = Math.sqrt(variance);
            
            let returnsMean = 0;
            let returnsStdDev = 0;
            
            if (returns.length > 0) {
                returnsMean = returns.reduce((a, b) => a + b, 0) / returns.length;
                const returnsVariance = returns.reduce((a, b) => a + Math.pow(b - returnsMean, 2), 0) / returns.length;
                returnsStdDev = Math.sqrt(returnsVariance);
            }
            
            const sortedPrices = [...prices].sort((a, b) => a - b);
            const median = sortedPrices.length % 2 === 0 
                ? (sortedPrices[sortedPrices.length / 2 - 1] + sortedPrices[sortedPrices.length / 2]) / 2
                : sortedPrices[Math.floor(sortedPrices.length / 2)];
            
            // Clear and populate statistics table
            $('#statsTable tbody').empty();
            
            const stats = [
                { name: 'Mean Price', value: `$${mean.toFixed(2)}` },
                { name: 'Median Price', value: `$${median.toFixed(2)}` },
                { name: 'Standard Deviation (Price)', value: `$${stdDev.toFixed(2)}` },
                { name: 'Coefficient of Variation', value: `${(stdDev / mean * 100).toFixed(2)}%` },
                { name: 'Mean Daily Return', value: `${returnsMean.toFixed(2)}%` },
                { name: 'Standard Deviation (Returns)', value: `${returnsStdDev.toFixed(2)}%` },
                { name: 'Sample Size', value: prices.length },
                { name: 'Date Range', value: `${data.dates[0]} to ${data.dates[data.dates.length - 1]}` }
            ];
            
            stats.forEach(stat => {
                $('#statsTable tbody').append(`
                    <tr>
                        <td><strong>${stat.name}</strong></td>
                        <td>${stat.value}</td>
                    </tr>
                `);
            });
            
        } catch (error) {
            console.error('Error updating statistics:', error);
            $('#statsTable tbody').html(`
                <tr>
                    <td colspan="2" class="text-center text-danger">
                        Error calculating statistics: ${error.message}
                    </td>
                </tr>
            `);
        }
    }
    
    // Run time series analysis
    $('#runTSAnalysisBtn').click(function() {
        if (!stockData) {
            alert('Please fetch stock data first');
            return;
        }
        
        // Show loading spinner - UPDATED
        $('.loading').addClass('show');
        $('#loadingText').html(`<span class="loading-dots">Running time series analysis</span>`);
        
        $.ajax({
            url: '/time_series_analysis',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                ticker: $('#tickerInput').val().trim().toUpperCase(),
                dates: stockData.dates,
                prices: stockData.prices
            }),
            success: function(response) {
                if (response.status === 'success') {
                    // Update plot images
                    $('#decompositionPlot').attr('src', 'data:image/png;base64,' + response.results.decomposition_plot);
                    $('#rollingStatsPlot').attr('src', 'data:image/png;base64,' + response.results.rolling_stats_plot);
                    $('#diffPlot').attr('src', 'data:image/png;base64,' + response.results.diff_plot);
                    $('#acfPacfPlot').attr('src', 'data:image/png;base64,' + response.results.acf_pacf_plot);
                    
                    // Display ADF test results
                    let adfHtml = '<table class="table table-sm">';
                    for (const [key, value] of Object.entries(response.results.adf_test)) {
                        adfHtml += `<tr><td>${key}</td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`;
                    }
                    adfHtml += '</table>';
                    $('#adfResults').html(adfHtml);
                    
                    let adfDiffHtml = '<table class="table table-sm">';
                    for (const [key, value] of Object.entries(response.results.adf_test_diff)) {
                        adfDiffHtml += `<tr><td>${key}</td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`;
                    }
                    adfDiffHtml += '</table>';
                    $('#adfDiffResults').html(adfDiffHtml);
                    
                    // Display stationarity results
                    if (response.results.is_stationary) {
                        $('#stationarityResult').html('<span class="badge badge-stationary">Stationary</span> <small>The series is stationary (p-value ≤ 0.05)</small>');
                    } else {
                        $('#stationarityResult').html('<span class="badge badge-non-stationary">Non-stationary</span> <small>The series is not stationary (p-value > 0.05)</small>');
                    }
                    
                    if (response.results.is_diff_stationary) {
                        $('#diffStationarityResult').html('<span class="badge badge-stationary">Stationary</span> <small>The differenced series is stationary (p-value ≤ 0.05)</small>');
                    } else {
                        $('#diffStationarityResult').html('<span class="badge badge-non-stationary">Non-stationary</span> <small>The differenced series is not stationary (p-value > 0.05)</small>');
                    }
                    
                    // Show analysis results
                    $('#tsAnalysisResults').show();
                } else {
                    alert(`Error: ${response.message}`);
                }
                
                // Hide loading spinner - UPDATED
                $('.loading').removeClass('show');
            },
            error: function(xhr, status, error) {
                $('.loading').removeClass('show'); // UPDATED
                alert('Error running time series analysis: ' + error);
            }
        });
    });
    
    // Run forecast model - Updated to handle both form submission and button click
    $('#forecastForm').on('submit', function(e) {
        e.preventDefault(); // Prevent default form submission
        e.stopPropagation(); // Stop event bubbling
        runForecast();
    });

    // Also handle direct button click (in case it's not in a form)
    $(document).on('click', 'button[type="submit"]', function(e) {
        if ($(this).closest('#forecastForm').length > 0) {
            e.preventDefault();
            e.stopPropagation();
            runForecast();
        }
    });

    // Extract forecast logic into a separate function
    function runForecast() {
        if (!stockData) {
            alert('Please fetch stock data first');
            return false;
        }

        const ticker = $('#tickerInput').val().trim().toUpperCase();
        const forecastDays = parseInt($('#forecastHorizon').val());
        const modelType = $('#forecastModel').val();

        console.log('Form values:', { ticker, forecastDays, modelType });

        if (!modelType || modelType === '') {
            alert('Please select a forecasting model');
            return false;
        }

        if (!forecastDays || forecastDays < 1 || forecastDays > 365) {
            alert('Please enter a valid forecast horizon between 1 and 365 days');
            return false;
        }

        // Show loading with better error handling - UPDATED
        try {
            $('.loading').addClass('show');
            $('#loadingText').html(`<span class="loading-dots">Running ${modelType.toUpperCase()} forecast model</span>`);
            
            console.log('Making AJAX request...');

            $.ajax({
                url: '/run_forecasting',
                type: 'POST',
                contentType: 'application/json',
                timeout: 120000,
                data: JSON.stringify({
                    ticker: ticker,
                    dates: stockData.dates,
                    prices: stockData.prices,
                    forecast_days: forecastDays,
                    model_type: modelType
                }),
                success: function(response) {
                    console.log('Forecast response received:', response);
                    
                    try {
                        if (response.status === 'success') {
                            const results = response.results;

                            // Update metrics safely
                            $('#mseMetric').text(results.metrics && results.metrics.mse ? results.metrics.mse.toFixed(2) : 'N/A');
                            $('#rmseMetric').text(results.metrics && results.metrics.rmse ? results.metrics.rmse.toFixed(2) : 'N/A');
                            $('#maeMetric').text(results.metrics && results.metrics.mae ? results.metrics.mae.toFixed(2) : 'N/A');

                            // Update plots safely
                            if (results.forecast_plot) {
                                $('#forecastPlotImg').attr('src', 'data:image/png;base64,' + results.forecast_plot);
                            }

                            if (results.backtest_plot) {
                                $('#backtestPlotImg').attr('src', 'data:image/png;base64,' + results.backtest_plot);
                            }

                            // Handle Prophet components
                            if (modelType === 'prophet' && results.components_plot) {
                                $('#componentsPlotImg').attr('src', 'data:image/png;base64,' + results.components_plot);
                                $('#componentsTabLi').show();
                            } else {
                                $('#componentsTabLi').hide();
                            }

                            // Populate forecast table safely
                            if (results.future_dates && results.forecast_values) {
                                $('#forecastTableData tbody').empty();
                                
                                for (let i = 0; i < results.future_dates.length; i++) {
                                    $('#forecastTableData tbody').append(`
                                        <tr>
                                            <td>${results.future_dates[i]}</td>
                                            <td>$${results.forecast_values[i].toFixed(2)}</td>
                                            <td>$${(results.lower_bound && results.lower_bound[i]) ? results.lower_bound[i].toFixed(2) : 'N/A'}</td>
                                            <td>$${(results.upper_bound && results.upper_bound[i]) ? results.upper_bound[i].toFixed(2) : 'N/A'}</td>
                                        </tr>
                                    `);
                                }
                            }

                            // Show results
                            $('#forecastResults').show();
                            $('#forecast-plot-tab').tab('show');

                        } else if (response.status === 'warning') {
                            const results = response.results;
                            
                            if (results.warning) {
                                alert('Warning: ' + results.warning);
                            }
                            
                            if (results.forecast_plot) {
                                $('#forecastPlotImg').attr('src', 'data:image/png;base64,' + results.forecast_plot);
                                $('#forecastResults').show();
                            }

                        } else {
                            alert(`Error: ${response.message || 'Unknown error occurred'}`);
                            if (response.traceback) {
                                console.error('Server traceback:', response.traceback);
                            }
                        }
                    } catch (processError) {
                        console.error('Error processing response:', processError);
                        alert('Error processing forecast results: ' + processError.message);
                    }
                },
                error: function(xhr, status, error) {
                    console.error('AJAX Error:', { xhr, status, error });
                    console.error('Response text:', xhr.responseText);
                    
                    let errorMessage = 'Error running forecast model: ';
                    if (status === 'timeout') {
                        errorMessage += 'Request timed out. Please try again.';
                    } else if (xhr.responseJSON && xhr.responseJSON.message) {
                        errorMessage += xhr.responseJSON.message;
                    } else {
                        errorMessage += error || 'Unknown error occurred';
                    }
                    
                    alert(errorMessage);
                },
                complete: function() {
                    // Always hide loading spinner - UPDATED
                    console.log('AJAX request completed, hiding loading...');
                    $('.loading').removeClass('show');
                }
            });
        } catch (ajaxError) {
            console.error('Error setting up AJAX request:', ajaxError);
            $('.loading').removeClass('show'); // UPDATED
            alert('Error setting up forecast request: ' + ajaxError.message);
        }

        return false;
    }

    // Model selection highlighting
    $('#modelType').change(function() {
        const selectedModel = $(this).val();
        $('.model-descriptions .card').removeClass('active');
        $(`#${selectedModel}Description`).addClass('active');
    });
    
    // Initialize with ARIMA highlighted
    $('#arimaDescription').addClass('active');

    // Model descriptions
    const modelDescriptions = {
        'arima': {
            title: 'ARIMA (AutoRegressive Integrated Moving Average)',
            description: 'ARIMA is a classical statistical method for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) components. Best suited for stationary time series with clear trends and patterns. Provides confidence intervals and is interpretable.',
            pros: 'Fast computation, interpretable results, good for short-term forecasts',
            cons: 'Requires stationary data, limited for complex non-linear patterns'
        },
        'prophet': {
            title: 'Facebook Prophet',
            description: 'Prophet is designed for forecasting time series with strong seasonal effects and several seasons of historical data. It handles missing data, outliers, and holiday effects automatically. Particularly good for business metrics with daily observations.',
            pros: 'Handles seasonality well, robust to outliers, automatic parameter tuning',
            cons: 'May overfit on small datasets, less suitable for high-frequency data'
        },
        'lstm': {
            title: 'LSTM (Long Short-Term Memory) Neural Network',
            description: 'LSTM is a type of recurrent neural network capable of learning long-term dependencies in sequential data. It can capture complex non-linear patterns and relationships in stock price movements. Requires more data but can handle complex patterns.',
            pros: 'Captures complex patterns, good for long sequences, handles non-linear relationships',
            cons: 'Requires large datasets, longer training time, less interpretable'
        }
    };

    // Handle model selection change
    $('#forecastModel').on('change', function() {
        const selectedModel = $(this).val();
        const $modelDescription = $('#modelDescription');
        const $modelDescriptionText = $('#modelDescriptionText');
        
        if (selectedModel && modelDescriptions[selectedModel]) {
            const model = modelDescriptions[selectedModel];
            
            // Create detailed description HTML
            const descriptionHTML = `
                <div class="model-info">
                    <h6 class="mb-2"><strong>${model.title}</strong></h6>
                    <p class="mb-2">${model.description}</p>
                    <div class="row">
                        <div class="col-md-6">
                            <p class="mb-1"><strong><i class="fas fa-check-circle text-success"></i> Advantages:</strong></p>
                            <p class="small text-muted">${model.pros}</p>
                        </div>
                        <div class="col-md-6">
                            <p class="mb-1"><strong><i class="fas fa-exclamation-triangle text-warning"></i> Limitations:</strong></p>
                            <p class="small text-muted">${model.cons}</p>
                        </div>
                    </div>
                </div>
            `;
            
            $modelDescriptionText.html(descriptionHTML);
            $modelDescription.removeClass('alert-info alert-warning alert-success');
            
            // Add different colors for different models
            if (selectedModel === 'arima') {
                $modelDescription.addClass('alert-info');
            } else if (selectedModel === 'prophet') {
                $modelDescription.addClass('alert-success');
            } else if (selectedModel === 'lstm') {
                $modelDescription.addClass('alert-warning');
            }
            
            $modelDescription.slideDown(300);
        } else {
            $modelDescription.slideUp(300);
        }
    });
});