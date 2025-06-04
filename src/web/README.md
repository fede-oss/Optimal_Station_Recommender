# Flask Web Application - ON HOLD ⚠️

## Status: Currently Not Working

**This Flask web application is currently on hold and is not functioning as it should.**

### Issues

The web interface for the Optimal Station Recommender is experiencing several issues:

- Configuration and import path conflicts between different modules
- Dependencies between the web application and the main data processing pipeline
- Incomplete integration with the updated project structure
- Potential conflicts in static file serving and routing

### Current State

- The Flask application files are present (`app.py`, `city_search.html`)
- The backend API endpoints are implemented but may not work correctly
- Static files and templates exist but routing may be broken
- The run script (`run_interactive_map.sh`) is available but the application may not start properly

### Recommendation

**Use the command-line interface instead:**

For generating station recommendations and interactive maps, please use the main pipeline scripts in the project root:

```bash
# Run the main data processing and analysis
python src/data/data_processing_refactored.py --city london

# Or use the simplified approach
python add_new_city.py
```

### Future Work

To restore the web application functionality, the following would need to be addressed:

1. **Fix import paths**: Resolve conflicts between different configuration modules
2. **Update dependencies**: Ensure all required data files and models are properly integrated
3. **Test routing**: Verify all Flask routes work with the current project structure
4. **Static file serving**: Fix any issues with serving HTML, CSS, and JavaScript files
5. **Error handling**: Improve error handling for missing data or processing failures

### Development Notes

- The web application was designed to provide an interactive interface for:
  - City selection and data processing
  - Real-time status updates during data fetching
  - Interactive map visualization of station recommendations
  - Comparison between predicted optimal locations and existing stations

---

**Last Updated**: June 2025  
**Status**: On Hold - Use CLI tools instead
