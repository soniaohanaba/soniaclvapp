// adding event listeners 

var countries_selector = document.getElementById("countries");
countries_selector.addEventListener("change", function(event){
	var country_index  = event.target.value;
	SELECTED_COUNTRY = COUNTRIES[country_index];
	console.log(SELECTED_COUNTRY);
	set_country_data(SELECTED_COUNTRY['name']);
});


var visualize_button = document.getElementById("visualize");
visualize_button.addEventListener("click", function(event){
	var selected_visualization = document.getElementById("visualization").value;
	var selected_graph = document.getElementById("graph_type").value;
	visualizations[selected_visualization]();
	display_data(selected_graph, GRAPH_DATA, TITLE);
	window.scrollTo(0,500);
})

function capitalize(text_string){
	return text_string[0].toUpperCase() + text_string.slice(1,);
}


function set_country_data(country){
	if (country == 'All'){
		COUNTRY_DATA = DATA;
		return COUNTRY_DATA;
	}
	COUNTRY_DATA = DATA.filter(function(data_point){
		if(data_point['country'] == country){
			return true;
		}
		return false;
	})
}

function clear_node(node){
	while(node.firstChild){
		node.removeChild(node.firstChild);
	}
	return node;
}

function createElement(tag_name, classes, innerHTML, dataset){
	var element = document.createElement(tag_name);
	element.className = classes;
	element.innerHTML = innerHTML;
	if(dataset){
		for(var property in dataset){
			element.dataset[property] = dataset[property];
		}
	}
	return element;
}

function add_select_options(data, select_tag, value_key, inner_html_key){
	var select_element = document.getElementById(select_tag);
	select_element = clear_node(select_element);
	for (var i = 0; i < data.length; i++){
		var option = document.createElement("option");
		option.value = data[i][value_key];
		option.innerHTML = capitalize(data[i][inner_html_key]);
		select_element.appendChild(option);
	}
}

function filter_countries(){
	var countries = [];
	for(var i = 0; i < DATA.length; i++){
		if(countries.indexOf(DATA[i]['country']) < 0){
			countries.push(DATA[i]['country']);
		}
	}
	countries = countries.map(function(country, index){
		return {
			name: country,
			index: index+1
		}
	})

	countries.unshift({name:'All', index:0})

	COUNTRIES = countries.slice(0,);
	console.log(COUNTRIES)
}
filter_countries()
filter_months()
add_select_options(COUNTRIES, "countries", "index", "name")




var visualizations = {
	month_vs_revenue: month_vs_revenue,
	month_vs_revenue_growth: month_vs_revenue_growth,
	quantity_vs_month: quantity_vs_month,
	month_vs_cust: month_vs_cust,
	month_vs_trans: month_vs_trans
}

function filter_months(){
	var months = [];
	for(var i = 0; i < DATA.length; i++){
		var invoice_date = new Date(DATA[i]['invoicedate']);
		var year_month = `${invoice_date.getFullYear()}-${invoice_date.getMonth()+1}`
		if (months.indexOf(year_month) < 0){
			months.push(year_month);
		}
	}
	console.log(months)
	MONTHS = months.sort(function(a,b){
		var a_date = (new Date(a)).getTime()
		var b_date = (new Date(b)).getTime()
		return a_date - b_date
	});
}

function display_table(x_axis_title, y_axis_title, data){
	var x_axis = document.getElementById('x_axis');
	x_axis.innerHTML = x_axis_title;
	var y_axis = document.getElementById('y_axis');
	y_axis.innerHTML = y_axis_title;
	var table__body	= document.getElementById("table__body");
	var table_container = document.getElementById("table_container");
	clear_node(table__body);
	table_container.classList.remove("hide");
	for(var i = 0; i < data.length; i++){
		var class_name = (i % 2 == 0) ? "table__row" : "table__row table__row--odd";
		var table__row = createElement("tr",class_name," ");
		var table_x_axis = createElement("td", "table__column", data[i]["x"]);
		var table_y_axis = createElement("td", "table__column", data[i]["y"]);
		table__body.appendChild(table__row);
		table__row.appendChild(table_x_axis);
		table__row.appendChild(table_y_axis);
	}
}

function month_vs_revenue(){
	var month_revenues = [];
	var data = (COUNTRY_DATA.length > 0 && COUNTRY_DATA) || DATA;
	console.log("data is ",data.length);
	for(var i = 0; i < MONTHS.length; i++){
		var month_revenue = 0;
		for(var u = 0; u < data.length; u++){
			if(data[u]['invoicedate'] == MONTHS[i]){
				var quantity = Number(data[u]['quantity'])
				var unit_price = Number(data[u]['unitprice'])
				// check if the quantity and unit prices are valid numbers
				quantity = quantity ? quantity : 0; // caters for 0 and NaN
				unit_price = unit_price ? unit_price : 0;
				var sale = quantity * unit_price;
				month_revenue += sale;
			}
		}
		var month_data = {
			"x":MONTHS[i],
			"y":month_revenue ? month_revenue : 'N/A'
		}
		month_revenues.push(month_data);
	}
	GRAPH_DATA = month_revenues;
	TITLE = "Monthly revenue";
	display_table("Month", "Revenue", GRAPH_DATA);
	return GRAPH_DATA;
}

function month_vs_revenue_growth(){
	var month_revenues = month_vs_revenue();
	var first_month = month_revenues[0];
	var first_month_y = first_month.y == 'N/A' ? 1 : first_month.y;
	var month_revenue_growth = month_revenues.map(function(month_revenue){
		var revenue_change = ((month_revenue.y - first_month_y)/first_month_y) * 100;
		return {
			'x': month_revenue['x'],
			'y': revenue_change
		}
	});
	GRAPH_DATA = month_revenue_growth;
	TITLE = "Percentage month revenue growth";
	display_table("Month", "Revenue growth", GRAPH_DATA);
	return GRAPH_DATA;
}

function quantity_vs_month(){
	var month_quantities = [];
	var data = ( (COUNTRY_DATA.length > 0) && COUNTRY_DATA) || DATA;
	for(var i = 0; i < MONTHS.length; i++){
		var month_quantity = 0;
		for(var u = 0; u < data.length; u++){
			if(data[u]['invoicedate'] = MONTHS[i]){
				var quantity = Number(data[u]['quantity'])
				// check if the quantity is a valid number
				quantity = quantity ? quantity : 0; // caters for 0 and NaN
				month_quantity += quantity;
			}
		}
		var month_data = {
			"x":MONTHS[i],
			"y":month_quantity ? month_quantity : 'N/A'
		}
		month_quantities.push(month_data);
	}
	GRAPH_DATA = month_quantities;
	TITLE = "Month product quantities bought";
	display_table("Month", "Quantity of products sold", GRAPH_DATA);
	return GRAPH_DATA;
}

function month_vs_cust(){
	var month_customers = [];
	var data = (COUNTRY_DATA.length > 0 && COUNTRY_DATA) || DATA;
	for(var i = 0; i < MONTHS.length; i++){
		var customers = [];
		for(var u = 0; u < data.length; u++){
			if(data[u]['invoicedate'] == MONTHS[i] && customers.indexOf(data[u]['customerid']) < 0){
				customers.push(data[u]['customerid'])
			}
		}
		var month_data = {
			"x":MONTHS[i],
			"y":customers.length
		}
		month_customers.push(month_data);
	}
	GRAPH_DATA = month_customers;
	TITLE = "Customers for the month";
	display_table("Month", "Number of customers", GRAPH_DATA);
	return GRAPH_DATA;
}

function month_vs_trans(){
	var month_transactions = [];
	var data = (COUNTRY_DATA.length > 0 && COUNTRY_DATA) || DATA;
	for(var i = 0; i < MONTHS.length; i++){
		var transactions = 0;
		for(var u = 0; u < data.length; u++){
			if(data[u]['invoicedate'] == MONTHS[i] ){
				transactions += 1;
			}
		}
		console.log(MONTHS[i], "month is")
		var month_data = {
			"x":MONTHS[i],
			"y":transactions
		}
		month_transactions.push(month_data);
	}
	GRAPH_DATA = month_transactions;
	TITLE = "Transactions for the month";
	display_table("Month", "Number of transactions", GRAPH_DATA);
	return GRAPH_DATA;
}

function display_data(graph_type, data, title){
	// clear the chart node
	var chart_container = document.getElementById("chart_container");
	chart_container = clear_node(chart_container);
	var new_canvas = document.createElement('canvas');
	new_canvas.id = "myChart1";
	new_canvas.width = "400";
	new_canvas.height = "200";
	chart_container.appendChild(new_canvas);
	var labels = GRAPH_DATA.map(function(point){
		return point.x;
	});
	var data = GRAPH_DATA.map(function(point){
		return point.y;
	});
	if (graph_type == 'bar'){
		display_bar_graph(labels, data, title);
	} else if(graph_type == 'line'){
		display_line_graph(labels, data, title);
	} else if(graph_type == 'scatter'){
		display_scatter_graph(GRAPH_DATA, title);
	}
}

function display_bar_graph(labels, data, title){
	var ctx = document.getElementById('myChart1');
	var background_colors = data.map(function(data){
		var red = Math.round(Math.random() * 255);
		var green = Math.round(Math.random() * 255);
		var blue = Math.round(Math.random() * 255);
		return `rgb(${red},${green},${blue},0.2)`;
	})
	var border_colors = data.map(function(data){
		var red = Math.round(Math.random() * 255);
		var green = Math.round(Math.random() * 255);
		var blue = Math.round(Math.random() * 255);
		return `rgb(${red},${green},${blue},1)`;
	})
	var myChart = new Chart(ctx, {
	    type: 'bar',
	    data: {
	        labels: labels,
	        datasets: [{
	            label: title,
	            data: data,
	            backgroundColor: background_colors,
	            borderColor: border_colors,
	            borderWidth: 1
	        }]
	    },
	    options: {
	        scales: {
	            yAxes: [{
	                ticks: {
	                    beginAtZero: true
	                }
	            }]
	        }
	    }
	});
}

function display_line_graph(labels, data, title){
	var ctx = document.getElementById('myChart1');
	var background_colors = data.map(function(data){
		var red = Math.round(Math.random() * 255);
		var green = Math.round(Math.random() * 255);
		var blue = Math.round(Math.random() * 255);
		return `rgb(${red},${green},${blue},0.2)`;
	})
	var border_colors = data.map(function(data){
		var red = Math.round(Math.random() * 255);
		var green = Math.round(Math.random() * 255);
		var blue = Math.round(Math.random() * 255);
		return `rgb(${red},${green},${blue},1)`;
	})
	var myChart = new Chart(ctx, {
	    type: 'line',
	    data: {
	        labels: labels,
	        datasets: [{
	            label: title,
	            data: data,
	            backgroundColor: background_colors,
	            borderColor: border_colors,
	            borderWidth: 1,
	            pointRadius: 5,
	        }]
	    },
	    options: {
	        scales: {
	            yAxes: [{
	                ticks: {
	                    beginAtZero: true
	                },
	                scaleLabel: {
	                	display: true,
	                	labelString: "frequency"
	                }
	            }]
	        }
	    }
	});
}