
class Path:
    def __init__(self):
        self.mpoi_path = []
        self.geo_list = []
        self.time_list = []
        self.density = None
        self.length = None
        self.quality = None
        self.popularity = None
        self.score = None
        self.trip_time = None

    def add_point(self, mpoi_id, geo, time):
        self.mpoi_path.append(mpoi_id)
        self.geo_list.append(geo)
        self.time_list.append(time)

    def update_trip_time(self):
        self.trip_time = self.time_list[-1] - self.time_list[0]

class Trip:
    def __init__(self):
        self.trip_time = None
        self.num_mpoi = 0
        self.mpoi_path = []
        self.geo_path = []
        self.photo_pmpoi = []
        self.quality = []
        self.time = []
        self.gain = None
        self.gain_list = []
        self.stay_time = []
        self.uid = None
        self.pids = []

    def add_photo(self, mpoi_id, geo_info, time_info, qual, pid):
        if not self.mpoi_path or self.mpoi_path[-1] != mpoi_id:
            self.num_mpoi += 1
            self.mpoi_path.append(mpoi_id)
            self.photo_pmpoi.append(1)
        else:
            self.photo_pmpoi[-1] += 1

        self.geo_path.append(geo_info)
        self.quality.append(qual)
        self.time.append(time_info)
        self.pids.append(pid)

    def update_trip_time(self):
        self.trip_time = self.time[-1] - self.time[0]

