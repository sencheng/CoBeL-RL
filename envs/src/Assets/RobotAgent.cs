using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using MLAgents.Sensors;
using MLAgents.SideChannels;

/*
 * Script for the robot agent 'GameObject'
 * 
 * is a attached to the root object of the RobotAgent-Prefab.
 * it implements the mlagents 'Agent' class by overriding the following methods:
 * 
 */
public class RobotAgent : Agent
{
    [SerializeField]
    public List<Wheel> m_wheel_list;

    [System.Serializable]
    public enum Axis
    {
        Front,
        Rear
    }

    [System.Serializable]
    public struct Wheel
    {
        public GameObject gameobject;
        public WheelCollider collider;
        public Axis axis;
    }

    public float m_max_motor_torgue;
    public float m_max_steer_angle;
    private float m_max_rotation_speed = 20f;
    public float m_torgue;
    public float m_steer_angle;
    private Rigidbody m_body;

    public GameObject m_target;
    private float m_reched_target_radius;
    private float m_spawn_target_distance;

    private Vector3 m_last_position;
    private float m_last_distance;

    FloatPropertiesChannel m_ResetParams;

    // Start is called before the first frame update
    void Start()
    {
        m_body = GetComponentInChildren<Rigidbody>();
        m_ResetParams = Academy.Instance.FloatProperties;
    }

    /*
     * 'MakeDecision' is called before each 'Academy' step.
     */

    public override void MakeDecision(int stepcount)
    {
        if(MaxStep(stepcount))
        {
            SetReward(-1f);
            Debug.LogError(stepcount + ": max step reached");
            // no decision is requested to avoid double obs
            return;
        }
        else
        {
            RequestDecision();
        }
    }

    public bool MaxStep(int stepcount)
    {
        return stepcount % maxStep == 0 && stepcount != 0;
    }

    public override void OnEpisodeBegin()
    {
        SetResetParameters();
        ResetRobot();
        ResetTarget();
        m_last_distance = GetDistanceToTarget();
        m_last_position = m_body.transform.position;
    }

    public void SetResetParameters()
    {
        m_reched_target_radius = m_ResetParams.GetPropertyWithDefault("target_reached_radius", 10);
        m_spawn_target_distance = m_ResetParams.GetPropertyWithDefault("target_spawn_distance", 25f);
        Debug.LogError("target_reached_radius = "+ m_reched_target_radius + ", m_spawn_target_distance = " + m_spawn_target_distance);
    }

    private void ResetRobot()
    {
        foreach (Wheel l_wheel in m_wheel_list)
        {
            l_wheel.collider.motorTorque = 0f;
            l_wheel.collider.steerAngle = 0f;
        }
        //reset transform
        m_body.transform.position = new Vector3(0f, 2f, 0f);
        m_body.transform.rotation = Quaternion.Euler(0f, Random.value*360f, 0f);
        //reset rigidbody
        m_body.velocity = Vector3.zero;
        m_body.angularVelocity = Vector3.zero;
    }

    private void ResetTarget()
    {
        m_target.transform.position = Vector3.up + Quaternion.Euler(0, Random.Range(0f, 360f), 0f) * Vector3.forward * m_spawn_target_distance;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        m_steer_angle = vectorAction[0];
        m_torgue = vectorAction[1];
    }

    public override float[] Heuristic()
    {
        var action = new float[2];

        action[0] = Input.GetAxis("Horizontal");
        action[1] = Input.GetAxis("Vertical");

        return action;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(GetDistanceToTarget());
        sensor.AddObservation((m_body.transform.position - m_target.transform.position).normalized);
        sensor.AddObservation(m_body.transform.forward);
    }

    private void GatherRewards()
    {
        // sparse rewards
        if (FellDown())
        {
            SetReward(-1f);
            EndEpisode();
            Debug.LogError(" dropped from plane :(");
        }
        else if (ReachedTarget())
        {
            SetReward(1f);
            EndEpisode();
            Debug.LogError(" reached it's target!");
        }
        // intermediate rewards
        else
        {

            if (IsCloser())
            {
                SetReward(-1f / (maxStep * 4));
            }
            else if (Moved())
            {
                SetReward(-1f / (maxStep * 2f));
            }
            else
            {
                //existence penalty
                SetReward(-1f / maxStep);
            }
            RequestDecision();
        }
    }

    public bool ReachedTarget()
    {
        float l_distance_to_target = GetDistanceToTarget();
        return l_distance_to_target < m_reched_target_radius;
    }

    private float GetDistanceToTarget()
    {
        return Vector3.Distance(m_body.transform.position, m_target.transform.position);
    }

    public bool FellDown()
    {
        return m_body.transform.position.y < -1f;
    }

    public bool Moved()
    {
        if (Vector3.Distance(m_last_position, m_body.transform.position) > 1f)
        {
            m_last_position = m_body.transform.position;
            return true;
        }
        else
        {
            return false;
        }
    }

    public bool IsCloser()
    {
        if (GetDistanceToTarget() < (m_last_distance))
        {
            m_last_distance = GetDistanceToTarget();
            return true;
        }
        else
        {
            return false;
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        foreach (Wheel l_wheel in m_wheel_list)
        {
            //visualize rotation
            l_wheel.gameobject.transform.parent.Rotate(m_max_rotation_speed * (m_body.velocity.z/10f), 0f, 0f, Space.Self);

            //apply steer angle to front axis.
            if (l_wheel.axis.Equals(Axis.Front))
            {
                float l_steer = m_max_steer_angle * m_steer_angle;
                l_wheel.collider.steerAngle = l_steer;

                //visualize steer angle
                l_wheel.gameobject.transform.parent.transform.parent.localRotation = Quaternion.Euler(0f, l_steer, 0f);
            }
            // apply motor force to rear axis.
            else
            {
                l_wheel.collider.motorTorque = m_max_motor_torgue * m_torgue;
            }
        }
        GatherRewards();
    }
}
