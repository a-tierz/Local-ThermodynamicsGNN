import argparse
import os

# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import mesh
from driverUtils import executeOnCaeStartup
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior


def todo_ASILMO(step_time=1.5, name_job='Job-1', v1=0.275, timeInterval=0.004, mesh_size=0.006):
    ## Property
    mdb.models['CocktailGlass_Water_2'].Material(name='Water')
    mdb.models['CocktailGlass_Water_2'].materials['Water'].Density(table=((983.2,),))
    mdb.models['CocktailGlass_Water_2'].materials['Water'].Viscosity(table=((0.0013,),))
    mdb.models['CocktailGlass_Water_2'].materials['Water'].Eos(type=USUP, table=((20.0,
                                                                                  0.0, 0.0),))
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    mdb.models['CocktailGlass_Water_2'].HomogeneousSolidSection(name='Section-1',
                                                                material='Water', thickness=None)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    region = p.Set(cells=cells, name='Set-Water')
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)

    ## Assembly
    a = mdb.models['CocktailGlass_Water_2'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    a.Instance(name='GLASS-1', part=p, dependent=ON)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    a.Instance(name='LIQUID-1', part=p, dependent=ON)

    ## Step
    mdb.models['CocktailGlass_Water_2'].ExplicitDynamicsStep(name='Step-1',
                                                             previous='Initial', timePeriod=step_time,
                                                             quadBulkViscosity=0.12)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')

    ## Interaction
    mdb.models['CocktailGlass_Water_2'].ContactProperty('IntProp-1')
    mdb.models['CocktailGlass_Water_2'].interactionProperties['IntProp-1'].TangentialBehavior(
        formulation=FRICTIONLESS)
    mdb.models['CocktailGlass_Water_2'].interactionProperties['IntProp-1'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    mdb.models['CocktailGlass_Water_2'].ContactExp(name='Int-1', createStepName='Step-1')
    mdb.models['CocktailGlass_Water_2'].interactions['Int-1'].includedPairs.setValuesInStep(
        stepName='Step-1', useAllstar=ON)
    mdb.models['CocktailGlass_Water_2'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'),))

    ## Load
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON,
                                                               predefinedFields=ON, connectors=ON,
                                                               adaptiveMeshConstraints=OFF)
    mdb.models['CocktailGlass_Water_2'].Gravity(name='Gravity', createStepName='Step-1',
                                                comp2=-9.8, distributionType=UNIFORM, field='')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    v2, e1, d2, n1 = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=v2[0])
    a1 = mdb.models['CocktailGlass_Water_2'].rootAssembly
    a1.regenerate()
    a = mdb.models['CocktailGlass_Water_2'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a = mdb.models['CocktailGlass_Water_2'].rootAssembly
    r1 = a.instances['GLASS-1'].referencePoints
    refPoints1 = (r1[2],)
    region = a.Set(referencePoints=refPoints1, name='Set-RP')
    mdb.models['CocktailGlass_Water_2'].EncastreBC(name='BC-1', createStepName='Initial',
                                                   region=region, localCsys=None)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    mdb.models['CocktailGlass_Water_2'].TabularAmplitude(name='Amp-1', timeSpan=STEP,
                                                         smooth=SOLVER_DEFAULT,
                                                         data=((0.0, 0.0), (0.1, 1.0), (0.2, 0.0), (1.0,
                                                                                                    0.0)))
    a = mdb.models['CocktailGlass_Water_2'].rootAssembly
    region = a.sets['Set-RP']
    mdb.models['CocktailGlass_Water_2'].VelocityBC(name='BC-2', createStepName='Step-1',
                                                   region=region, v1=0.0, v2=0.0, v3=v1, vr1=0.0, vr2=0.0, vr3=0.0,
                                                   amplitude='Amp-1', localCsys=None, distributionType=UNIFORM,
                                                   fieldName='')
    mdb.models['CocktailGlass_Water_2'].boundaryConditions['BC-1'].deactivate('Step-1')

    ## Mesh
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    p.seedPart(size=mesh_size*2, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    p.generateMesh()
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    p.generateMesh()
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                              kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF,
                              hourglassControl=DEFAULT, distortionControl=DEFAULT,
                              particleConversion=TIME, particleConversionThreshold=0.0,
                              particleConversionPPD=1, particleConversionKernel=CUBIC)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    pickedRegions = (cells,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2,
                                                       elemType3))

    ## Output history
    mdb.models['CocktailGlass_Water_2'].fieldOutputRequests['F-Output-1'].setValues(
        variables=('COORD', 'U', 'V', 'ELEN', 'ENER', 'PEVAVG', 'VE', 'S', 'ELVD'),
        timeInterval=timeInterval)
    mdb.models['CocktailGlass_Water_2'].historyOutputRequests['H-Output-1'].setValues(
        variables=('ETOTAL',), timeInterval=timeInterval)

    ## Create job
    a = mdb.models['CocktailGlass_Water_2'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    mdb.Job(name=name_job, model='CocktailGlass_Water_2')
    mdb.jobs[name_job].submit()
    # mdb.jobs[name_job].writeNastranInputFile()


def Parts(r1, r2, h1, h2, x):
    mdb.Model(name='CocktailGlass_Water_2', modelType=STANDARD_EXPLICIT)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)

    s = mdb.models['CocktailGlass_Water_2'].ConstrainedSketch(name='__profile__',
                                                              sheetSize=0.02)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.sketchOptions.setValues(decimalPlaces=4)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -0.01), point2=(0.0, 0.01))
    s.FixedConstraint(entity=g[2])
    s.Line(point1=(0.0, 0.0), point2=(r1, 0.0))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.Line(point1=(r1, 0.0), point2=(r2, h2))
    session.viewports['Viewport: 1'].view.fitView()
    p = mdb.models['CocktailGlass_Water_2'].Part(name='GLASS',
                                                 dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    p.BaseShellRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
    s.unsetPrimaryObject()
    p = mdb.models['CocktailGlass_Water_2'].parts['GLASS']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['CocktailGlass_Water_2'].sketches['__profile__']
    s1 = mdb.models['CocktailGlass_Water_2'].ConstrainedSketch(name='__profile__',
                                                               sheetSize=0.2)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.sketchOptions.setValues(decimalPlaces=3)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -0.1), point2=(0.0, 0.1))
    s1.FixedConstraint(entity=g[2])
    s1.Line(point1=(0.0, 0.0), point2=(r1, 0.0))
    s1.HorizontalConstraint(entity=g[3], addUndoState=False)
    s1.Line(point1=(r1, 0.0), point2=(x, h1))
    s1.Line(point1=(x, h1), point2=(0.0, h1))
    s1.HorizontalConstraint(entity=g[5], addUndoState=False)
    s1.Line(point1=(0.0, h1), point2=(0.0, 0.0))
    s1.VerticalConstraint(entity=g[6], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    p = mdb.models['CocktailGlass_Water_2'].Part(name='LIQUID',
                                                 dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    p.BaseSolidRevolve(sketch=s1, angle=360.0, flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()
    p = mdb.models['CocktailGlass_Water_2'].parts['LIQUID']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['CocktailGlass_Water_2'].sketches['__profile__']


def generate_report(name_job):
    # Create a viewport and set it as the current one
    session.Viewport(name='Viewport: 1',
                     origin=(0.0, 0.0),
                     width=253.28515625,
                     height=127.874992370605)
    session.viewports['Viewport: 1'].makeCurrent()
    session.viewports['Viewport: 1'].maximize()
    executeOnCaeStartup()

    odb_file = name_job + '.odb'
    odb_session = session.openOdb(name=odb_file)
    # odb_session = session.openOdb(name = odb_file + '.odb')

    session.viewports['Viewport: 1'].setValues(displayedObject=odb_session)

    odb = session.odbs[odb_file]

    # Extracting Step 1, this analysis only had one step
    step1 = odb_session.steps.values()[0]

    # Loop over all frames in Step 1
    for count, i in enumerate(odb_session.steps[step1.name].frames):
        # Set the frame to display in the viewport
        session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=count)
        # session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(computeOrder=EXTRAPOLATE_AVERAGE_COMPUTE)

        # Configure field report options
        session.fieldReportOptions.setValues(printTotal=OFF,
                                             printMinMax=OFF, )
        # numberFormat=nf)

        # Write a field report to a text file
        session.writeFieldReport(
            fileName=os.path.splitext(odb_file)[0] + '_variables_data.txt',
            append=ON,
            sortItem='Element Label',
            odb=odb,
            step=0,
            frame=count,
            outputPosition=NODAL,
            variable=(
                ('COORD', NODAL),
                ('V', NODAL),  # Nodal displacements
                ('ELVD', WHOLE_ELEMENT),
                # ('S', INTEGRATION_POINT)  # Nodal tension
            ))
    odb.close()



def main():
    file_name = os.listdir(os.getcwd())[2]
    print(file_name)

    # file_name = 'Glass_2_5_2_5.txt'
    with open(file_name, "r") as file:
        params = file.read()
    lines_params = params.split('\n')

    print(lines_params[0])
    r1 = float(lines_params[0])
    r2 = float(lines_params[1])
    h1 = float(lines_params[2])
    h2 = float(lines_params[3])
    v1 = float(lines_params[4])

    dr = r2 - r1
    x = h1 * dr / h2 + r1
    x = round(x, 4)

    #
    Parts(r1, r2, h1, h2, x)
    name_job = file_name[:-4]
    #todo_ASILMO(name_job=name_job, step_time=1.5, timeInterval=0.012, v1=v1, mesh_size=0.0045)
    todo_ASILMO(name_job=name_job, step_time=1, timeInterval=0.00833, v1=v1, mesh_size=0.0035)

    generate_report(name_job)


if __name__ == "__main__":
    main()
