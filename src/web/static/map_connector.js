// map_connector.js - Connects the city search interface with the interactive map
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're inside an iframe
    const isInIframe = window.location !== window.parent.location;
    
    // Function to extract query parameters from URL
    function getQueryParam(param) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(param);
    }
    
    // Check if a city parameter was passed
    const city = getQueryParam('city');
    if (city) {
        console.log(`City parameter detected: ${city}`);
        // Send request to get the city data
        fetchCityData(city);
    }
    
    // Function to fetch city data and update the map
    function fetchCityData(cityName) {
        console.log(`Fetching data for city: ${cityName}`);
        
        // Create a status element
        let statusElement = document.createElement('div');
        statusElement.id = 'cityStatus';
        statusElement.style.cssText = 'position: fixed; top: 60px; left: 50px; background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; z-index: 9999; font-size: 14px;';
        statusElement.innerHTML = `Processing data for ${cityName}...`;
        document.body.appendChild(statusElement);
        
        fetch('/predict_city', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ city: cityName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                statusElement.style.backgroundColor = '#f8d7da';
                statusElement.style.color = '#721c24';
                statusElement.innerHTML = `Error: ${data.detail}`;
                
                // Make status disappear after 10 seconds
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 10000);
            } else {
                statusElement.style.backgroundColor = '#d4edda';
                statusElement.style.color = '#155724';
                statusElement.innerHTML = `Successfully loaded data for ${cityName}!`;
                
                // Make status disappear after 5 seconds
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 5000);
                
                // Update the map with the received data
                updateMapWithCityData(data);
            }
        })
        .catch(error => {
            console.error('Error fetching city data:', error);
            statusElement.style.backgroundColor = '#f8d7da';
            statusElement.style.color = '#721c24';
            statusElement.innerHTML = 'An error occurred while fetching data';
            
            // Make status disappear after 10 seconds
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 10000);
        });
    }
    
    // Function to update the map with city data
    function updateMapWithCityData(data) {
        console.log('Updating map with city data');
        
        // Find the map layers based on the specific GeoJSON add functions in the page
        // This requires knowledge of how the interactive map was generated
        try {
            // Add predictions layer if that function exists
            if (typeof window.geo_json_3326840883bfc847d92f90d2c44b5e09_add === 'function') {
                window.geo_json_3326840883bfc847d92f90d2c44b5e09_add(data.prediction_h3_geojson);
                console.log('Added prediction layer');
            }
            
            // Add stations layer if that function exists
            if (typeof window.geo_json_b656a5a35467cb3fc30420e061df951c_add === 'function') {
                window.geo_json_b656a5a35467cb3fc30420e061df951c_add(data.actual_stations_geojson);
                console.log('Added stations layer');
            }
            
            // Add boundary layer if that function exists
            // Note: The exact function name for boundary might be different
            const functionNames = Object.keys(window).filter(name => 
                name.startsWith('geo_json_') && name.includes('add') && 
                name !== 'geo_json_3326840883bfc847d92f90d2c44b5e09_add' && 
                name !== 'geo_json_b656a5a35467cb3fc30420e061df951c_add'
            );
            
            if (functionNames.length > 0) {
                // Try to add boundary with the first available add function
                window[functionNames[0]](data.city_boundary_geojson);
                console.log('Added boundary layer using', functionNames[0]);
            }
            
            // Get the map object (assumes it's the first Leaflet map in the page)
            const mapObjects = Object.values(window).filter(obj => obj instanceof L.Map);
            if (mapObjects.length > 0) {
                // Fit map to the boundary
                const features = data.city_boundary_geojson.features;
                if (features && features.length > 0) {
                    const bounds = L.geoJSON(features).getBounds();
                    mapObjects[0].fitBounds(bounds);
                    console.log('Fit map to city bounds');
                }
            }
        } catch (e) {
            console.error('Error updating map:', e);
        }
    }
});
