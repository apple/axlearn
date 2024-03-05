from ml_goodput_measurement import goodput

def main():
    run_name='{config.run_name}'
    goodput_logger_name = f'goodput_{run_name}'
    # Create a GoodPut Calculator object
    goodput_calculator = goodput.GoodputCalculator(job_name=run_name, logger_name=goodput_logger_name)

    # TODO: make this a loop to periodically pull Goodput
    current_goodput = goodput_calculator.get_job_goodput()
    print(f"=========> Current job goodput: {current_goodput:.2f}%")

if __name__ == "__main__":
    main()