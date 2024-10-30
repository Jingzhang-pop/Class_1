#Task 1
function odd_or_even(n)
    if n % 2 ==0
        println("even")
        else
            println("odd")
        end
    end 

#Task 2

function compare_three(a, b, c)
    if a > 0 && b > 0 && c > 0
        println("All number are positive")
    elseif a < 0 || b < 0 || c < 0
        println("At least one number is not positive")
    else a == 0 && b == 0 && c == 0
        println("All number are zero")
    end
end

#Task 3

function my_factorial(n::Int)
    result = 1
    for i in 1:n
        result *= i
    end
    return result
end


#Task 4

function count_positives(arr::Array{Int})
    count = 0
    for num in arr
        if num > 0
            count += 1
        end
    end
    return count
end

using Pkg
Pkg.add("Plots")


#Task 5.1
using Plots
function plot_powers(n)
    power_plots = plot(title="Powers of x", xlabel="x", ylabel="y")
    for i in 1:n
        f(x)=x^i
        plot!(-10:0.2:10, f, label="x^$i", lw=3)
    end
    display(power_plots)
    
end

plot_powers(3)

#Task 5.2
function count_positives_broadcasting(arr)
    positive_mask = arr .> 0
    return sum(positive_mask)
end
count_positives_broadcasting([1, -3, 4, 7, -2, 0])
count_positives_broadcasting([-5, -10, 0, 6]
)

#Task 6
function standard_deviation(x)
    mean_x = sum(x)/length(x)
    squared_d = (x.-mean_x).^2
    variance = sum(squared_d)/(length(x)-1)
    return sqrt(variance)
end 
standard_deviation([1,2,3,4,5])
standard_deviation([5,10,15])
standard_deviation(collect(2:7))



#Task 7
using DelimitedFiles, Plots, Statistics
data = readdlm("/Users/mac/Downloads/dataset.csv", ',', Float64)
earnings = data[:, 1]
education = data[:, 2]
hours_worked = data[:,3]
scatter(education, earnings, color="green", xlabel="Education Level", ylabel="Earnings", title="Earnings and Education Level", label="Earnings and eduvation")
scatter!(hours_worked, earnings, color="red", xlabel="hours_worked per week", ylabel="Earnings", title="Earnings and hours worked", label="earnings and hours worked")
corr_earnings_education = cor(earnings, education)
corr_earning_hours = cor(earnings, hours_worked)
println("Correlation between earnings and edncation level: ", corr_earnings_education)
println("Correlation between earnings and hours worked: ", corr_earning_hours)
println("The scatter plots and Correlation values indicate the relationships between earnings and the variable")