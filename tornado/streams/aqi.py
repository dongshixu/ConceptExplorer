""" AQI(Air quality index) class.
"""
class AQI(object):
    IAQI = [0, 50, 100, 150, 200, 300, 400, 500]
    CP = {
        'SO2':[0, 150, 500, 650, 800, 1600, 2100, 2620],
        'NO2':[0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'CO': [0, 5000, 10000, 35000, 60000, 90000, 120000, 150000],
        'O3': [0, 160, 200, 300, 400, 800, 1000, 1200],
        'O3_8hours': [0, 100, 160, 215, 265, 800, 1000, 1200],
        'PM2.5_day': [0, 35, 75, 115, 150, 250, 350, 500],
        'PM10_day': [0, 50, 150, 250, 350, 420, 500, 600],
    }
    POLLUTANT = ['SO2', 'NO2', 'CO', 'O3', 'O3_8hours', 'PM2.5_day', 'PM10_day']

    @staticmethod
    def IAQI_P(p, cp):
        """ Return IAQI of any pollutant [SO2, NO2, C0, O3, PM2.5, PM10]

            @params:
            p: pollutant type, belong to [SO2, NO2, C0, O3, PM2.5, PM10]
            cp: concentration of pollutant p

            @return:
            IAQI_P: IAQI of pollutant p

        """
        flag = False
        range_index = 0
        CP_list = AQI.CP[p]
        for i, v in enumerate(CP_list):
            if cp < v:
                range_index = i
                flag = True
                break

        if not flag:
            return AQI.IAQI[-1]

        BP_l = CP_list[range_index - 1]
        BP_h = CP_list[range_index]
        IAQI_l = AQI.IAQI[range_index - 1]
        IAQI_h = AQI.IAQI[range_index]
        return (IAQI_h - IAQI_l) / (BP_h - BP_l) * (cp - BP_l) + IAQI_l
