import subprocess
import os
import numpy as np
import time


def clean_artifact_files(directory):
    # Define the allowed file extensions
    allowed_extensions = ['.odb', '.inp', '.txt', '.py', '.cae']

    # Iterate through files in the directory and delete files with disallowed extensions
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension not in allowed_extensions:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')


def run_abaqus_script(script_path):
    num_samples = 15

    cnt1 = list(np.round(np.ones((num_samples))*0.045, 3)) # r1
    cnt1 = np.round(np.random.uniform(0.02, 0.04, num_samples), 3)  # r1
    cnt2 = np.round(np.random.uniform(0.04, 0.06, num_samples), 3)  # r2
    cnt3 = np.round(np.random.uniform(0.03, 0.05, num_samples), 3)  # h1
    cnt4 = np.round(np.random.uniform(0.09, 0.11, num_samples), 3)  # h2
    vel1 = np.round(np.random.uniform(0.25, 0.4, int(num_samples / 2)), 3)  # vel
    vel2 = np.round(np.random.uniform(-0.25, -0.4, int(num_samples / 2)), 3)  # vel
    vel = np.concatenate((vel1, vel2))

    # Test case
    # cnt1 = [0.0275]  # r1
    # cnt2 = [0.0425]  # r2
    # cnt3 = [0.046]  # h1
    # cnt4 = [0.115]  # h2
    # vel = [0.35]

    names = []
    cwd = os.getcwd()

    for i in range(len(cnt1)):
        name = f'GlassV4_m0035_{(int(cnt1[i] * 1000))}_{(int(cnt2[i] * 1000))}_{(int(cnt3[i] * 1000))}_{(int(cnt4[i] * 1000))}_{(int(vel[i] * 1000))}'
        print(name)
        names.append(name)

        output_dir = os.path.join(cwd, 'data', name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            continue  #Quitar si queremos que sobreescriba!!!!
        os.chdir(output_dir)

        with open(name + '.txt', 'w') as archivo:
            archivo.write(str(cnt1[i]) + '\n')
            archivo.write(str(cnt2[i]) + '\n')
            archivo.write(str(cnt3[i]) + '\n')
            archivo.write(str(cnt4[i]) + '\n')
            archivo.write(str(vel[i]) + '\n')

        command = f'abaqus cae noGUI={os.path.join(cwd, script_path)}'
        try:
            start = time.time()
            print(f"Script starts at: { start}")
            # Ejecuta el comando
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Script executed successfully. in {(time.time()-start)/60} min")
            print("Output:", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print("Error executing script.")
            print("Output:", e.stdout.decode())
            print("Error:", e.stderr.decode())

        clean_artifact_files(output_dir)

        os.chdir(cwd)


if __name__ == "__main__":
    # Ruta al archivo de macros de Abaqus
    script_path = r'abaqusMacros.py'
    run_abaqus_script(script_path)
