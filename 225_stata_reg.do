clear
set more off
capture log close
log using "225_finalpj.log"
use get_it_done_2018.dta
save get_it_done_2018_YH, replace



*cleaning data
keep if status == "Closed"
save get_it_done_2018_YH, replace
tab service_name, sort


gen top5 = 1 if service_name == "72 Hour Violation" 
replace top5 = 2 if service_name == "Graffiti Removal" 
replace top5 = 3 if service_name == "Pothole" 
replace top5 = 4 if service_name == "Missed Collection" 
replace top5 = 5 if service_name == "Illegal Dumping"
drop if top5 ==.
sort top5
preserve
gen case_age_days_sd = case_age_days
collapse (count) latutude (mean) case_age_days (sd) case_age_days_sd, by(service_name)
li
restore


*data speculation
tabulate top5, generate(rank)
graph bar (mean)rank*, over(district) stack title("Proportion of Top 5 service, by District") ///
legend(label(1 "72 Hour Violation") label(2 "Graffiti Removal") label(3 "Pothole") label(4 "Missed Collection") label(5 "Illegal Dumping")) ///
ytitle("Proportion of service requested") 
graph export service_proportion.png, replace

graph bar (mean)case_age_days, over(district, label(angle(45))) title("Average duration of Top 5 service by district") /// 
ytitle("Duration(days)") legend(title("Districts")) asyvar
graph export caseage_district.png, replace



label define serlab 1 "72 Hour Violation" 2 "Graffiti Removal" 3 "Pothole" ///
4 "Missed Collection"  5 "Illegal Dumping"
label values top5 serlab



*regression

gen afram_p = afram_pop2017/total_pop2017
gen white_p = white_pop2017/total_pop2017
gen hisp_p = hisp_pop2017/total_pop2017
eststo clear
eststo: reg case_age_days i.district avg_agi i.top5 afram_p white_p hisp_p
eststo: reg case_age_days i.top5##i.district
eststo: reg case_age_days c.avg_agi##i.district
esttab *, se ar2 scalars(rmse) 
esttab * using reg3-1b, rtf replace se ar2 scalars(rmse)
